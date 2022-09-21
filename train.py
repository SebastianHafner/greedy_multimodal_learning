#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gin
from gin.config import _CONFIG
import torch
import logging

from src import dataset
from src import callbacks as avail_callbacks
from src.model import MMTM_DSUNet
from src.utils import gin_wrap
from src.loss_function import blend_loss
from src.metric import f1
from src.framework import Framework

logger = logging.getLogger(__name__)


@gin.configurable
def train(config, dataset_path, save_path, lr, wd, batch_size, n_epochs, nummodalities, seed, other_callbacks: list):

    _CONFIG['name'] = config

    # make training deterministic
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = MMTM_DSUNet()
    model = torch.nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Sending model to {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    metrics = [f1]

    framework = Framework(
        model=model,
        optimizer=optimizer,
        loss_function=blend_loss,
        metrics=metrics,
        verbose=True,
        nummodalities=nummodalities,
        config=_CONFIG,
        device=device,
    )

    train, valid, test = dataset.get_urbanmappingdata(dataset_path, batch_size=batch_size, seed=seed)

    callbacks = []
    for name in other_callbacks:
        if name in avail_callbacks.__dict__:
            clbk = avail_callbacks.__dict__[name]()
            callbacks.append(clbk)

    # default callbacks
    callbacks.append(avail_callbacks.ModelCheckpoint(save_path, config))
    callbacks.append(avail_callbacks.ProgressionCallback())
    callbacks.append(avail_callbacks.WBLoggingCallback(metrics=[m.__name__ for m in metrics], run_name=config))

    # Configure callbacks
    for clbk in callbacks:
        clbk.set_save_path(save_path)
        clbk.set_model(model, ignore=False)
        clbk.set_optimizer(optimizer)
        clbk.set_config(config)
        clbk.set_model_pytoune(framework)

    callback_list = avail_callbacks.CallbackList(callbacks)
    callback_list.set_params({'epochs': n_epochs, 'steps': len(train)})

    _ = framework.train_loop(train, valid_generator=valid, test_generator=test, epochs=n_epochs,
                             callback_list=callback_list)


if __name__ == "__main__":
    gin_wrap(train)
