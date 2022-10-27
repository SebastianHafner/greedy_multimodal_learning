#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gin
from gin.config import _CONFIG
import logging
from src import dataset
from src import callbacks as avail_callbacks 
from src.model import MMTM_DSUNet
from src.utils import gin_wrap, save_weights
from src.metric import f1, tp, fp, fn
from src.framework import Framework
import torch
from pathlib import Path

from train import blend_loss

logger = logging.getLogger(__name__)


def load_pretrained_model(model, optimizer, save_path):
    checkpoint = torch.load(save_path)
    model_dict = model.state_dict()
    model_dict.update(checkpoint['model'])
    model.load_state_dict(model_dict, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.info("Done reloading!")
    return model, optimizer


@gin.configurable
def eval(config, dataset_path, save_path, batch_size=4, nummodalities=2, model_type='end_training'):

    _CONFIG['name'] = config

    model = MMTM_DSUNet()
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters())
    pretrained_weights_path = Path(save_path) / 'networks' / f'model_{model_type}_{config}.pt'
    model, optimizer = load_pretrained_model(model, optimizer, pretrained_weights_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Sending model to {device}")

    metrics = [tp, fp, fn]

    framework = Framework(
        model=model,
        optimizer=None,
        loss_function=blend_loss,
        metrics=metrics,
        nummodalities=nummodalities,
        config=_CONFIG,
        device=device
    )

    train, val, test = dataset.get_urbanmappingdata(dataset_path, batch_size=batch_size)

    framework.inference_loop(test, 'test', save_path)

    if not model.module.mmtm_squeeze_features_recorded():
        callback = avail_callbacks.EvalProgressionCallback(phase='train', steps=len(train))
        model = framework.record_mmtm_features(train, avail_callbacks.CallbackList([]))
        file_name = Path(save_path) / 'networks' / f'model_{model_type}_{config}.pt'
        save_weights(model, optimizer, file_name)

    framework.eval_loop(test, 'test', save_path)


if __name__ == "__main__":
    gin_wrap(eval)
