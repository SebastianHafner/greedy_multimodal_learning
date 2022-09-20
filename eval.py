#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gin
from gin.config import _CONFIG
import logging
from src import dataset
from src import callbacks as avail_callbacks 
from src.model import MMTM_DSUNet
from src.utils import gin_wrap
from src.metric import f1
import torch

from train import blend_loss

logger = logging.getLogger(__name__)


@gin.configurable
def eval_(config, dataset_path, save_path, batch_size=4, callbacks=[]):

    _CONFIG['name'] = config

    model = MMTM_DSUNet()
    model = torch.nn.DataParallel(model)
    train, val, test = dataset.get_urbanmappingdata(dataset_path, batch_size=batch_size)

    # Create dynamically callbacks
    callbacks_constructed = []
    for name in callbacks:
        if name in avail_callbacks.__dict__:
            clbk = avail_callbacks.__dict__[name]()
            callbacks_constructed.append(clbk)

    mmtm_features_available = False
    if not mmtm_features_available:
        model.module.saving_mmtm_squeeze_array = True
        evalution_loop(model=model, loss_function=blend_loss, metrics=[f1], config=_CONFIG, save_path=save_path,
                       test=train, test_steps=len(train), custom_callbacks=callbacks_constructed)

    model.module.saving_mmtm_squeeze_array = False
    model.module.mmtm_off = True
    evalution_loop(model=model,  loss_function=blend_loss, metrics=[f1], config=_CONFIG,  save_path=save_path,
                   test=test, test_steps=len(test), custom_callbacks=callbacks_constructed)


def _load_pretrained_model(model, save_path):
    checkpoint = torch.load(save_path)
    model_dict = model.state_dict()
    model_dict.update(checkpoint['model'])
    model.load_state_dict(model_dict, strict=False)
    logger.info("Done reloading!")


@gin.configurable
def evalution_loop(model, loss_function, metrics, config, save_path, test=None, test_steps=None, custom_callbacks=[],
                   nummodalities=2, ):
    pretrained_weights_path = Path(save_path) / 'networks' / f'model_best_val_{config["name"]}.pt'
    _load_pretrained_model(model, pretrained_weights_path)

    callbacks = list(custom_callbacks)

    # Configure callbacks
    for clbk in callbacks:
        clbk.set_save_path(save_path)
        clbk.set_model(model, ignore=False)  # TODO: Remove this trick
        clbk.set_config(config)

    model = Model_(
        model=model,
        optimizer=None,
        loss_function=loss_function,
        metrics=metrics,
        nummodalities=nummodalities,
        config=config,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Sending model to {device}")

    model.eval_loop(
        test,
        epochs=0,
        test_steps=test_steps,
        callbacks=callbacks
    )


if __name__ == "__main__":
    gin_wrap(eval_)
