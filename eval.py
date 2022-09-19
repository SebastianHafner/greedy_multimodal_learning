#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gin
from gin.config import _CONFIG
import logging
from src import dataset
from src import callbacks as avail_callbacks 
from src.model import MMTM_DSUNet
from src.training_loop import evalution_loop
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


if __name__ == "__main__":
    gin_wrap(eval_)
