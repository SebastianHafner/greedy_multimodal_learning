#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gin
from gin.config import _CONFIG
import torch
import logging

from src import dataset
from src import callbacks as avail_callbacks 
from src.model import MMTM_DSUNet
from src.training_loop import training_loop
from src.utils import gin_wrap
from src.loss_function import power_jaccard_loss
from src.metric import f1

logger = logging.getLogger(__name__)


def blend_loss(y_hat, y):
    loss_func = power_jaccard_loss
    losses = []
    for y_pred in y_hat:
        losses.append(loss_func(y_pred, y))

    return sum(losses)


@gin.configurable
def train(save_path, wd, lr, momentum, batch_size, callbacks=[]):

    model = MMTM_DSUNet()
    train, valid, test = dataset.get_urbanmappingdata(batch_size=batch_size)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr, 
        weight_decay=wd, 
        momentum=momentum
    )

    callbacks_constructed = []
    for name in callbacks:
        if name in avail_callbacks.__dict__:
            clbk = avail_callbacks.__dict__[name]()
            callbacks_constructed.append(clbk)

    training_loop(
        model=model,
        optimizer=optimizer, 
        loss_function=blend_loss, 
        metrics=[f1],
        train=train, valid=valid, test=test, 
        steps_per_epoch=len(train),
        validation_steps=len(valid),
        test_steps=len(test),
        save_path=save_path, 
        config=_CONFIG,
        custom_callbacks=callbacks_constructed
    )


if __name__ == "__main__":
    gin_wrap(train)
