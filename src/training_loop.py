# -*- coding: utf-8 -*-
"""
A gorgeous, self-contained, training loop. Uses Poutyne implementation, but this can be swapped later.
"""

import logging
import os
import tqdm
import pickle
from functools import partial

import numpy as np
import pandas as pd
import torch
import gin

from src.callbacks import ModelCheckpoint, LambdaCallback
from src.utils import save_weights
from src.framework import Model_

logger = logging.getLogger(__name__)


def _construct_default_callbacks(model, optimizer, save_path, checkpoint_monitor):
    callbacks = []

    callbacks.append(
        ModelCheckpoint(
            monitor=checkpoint_monitor,
            save_best_only=True,
            mode='max',
            filepath=os.path.join(save_path, "model_best_val.pt"))
    )
    
    def save_weights_fnc(epoch, logs):
        logger.info("Saving model from epoch " + str(epoch))
        save_weights(model, optimizer, os.path.join(save_path, "model_last_epoch.pt"))

    callbacks.append(LambdaCallback(on_epoch_end=save_weights_fnc))

    return callbacks


def _load_pretrained_model(model, save_path):
    checkpoint = torch.load(save_path)
    model_dict = model.state_dict()
    model_dict.update(checkpoint['model']) 
    model.load_state_dict(model_dict, strict=False)
    logger.info("Done reloading!")


@gin.configurable
def training_loop(model, loss_function, metrics, optimizer, config, save_path,  steps_per_epoch, train=None,
                  valid=None, test=None, test_steps=None, validation_steps=None, use_gpu=False, device_numbers=[0],
                  custom_callbacks=[], checkpoint_monitor="val_f1", n_epochs=100, verbose=True, nummodalities=2):

    callbacks = list(custom_callbacks)

    callbacks += _construct_default_callbacks(model, optimizer, save_path, checkpoint_monitor)
    
    # Configure callbacks
    for clbk in callbacks:
        clbk.set_save_path(save_path)
        clbk.set_model(model, ignore=False)  # TODO: Remove this trick
        clbk.set_optimizer(optimizer)
        clbk.set_config(config)

    model = Model_(
        model=model,
        optimizer=optimizer, 
        loss_function=loss_function, 
        metrics=metrics,
        verbose=verbose,
        nummodalities=nummodalities,
    )
            
    for clbk in callbacks:
        clbk.set_model_pytoune(model)

    if use_gpu and torch.cuda.is_available(): 
        base_device = torch.device("cuda:{}".format(device_numbers[0]))
        model.to(base_device)
        logger.info("Sending model to {}".format(base_device))
        
    _ = model.train_loop(
        train,
        valid_generator=valid,
        test_generator=test,
        test_steps=test_steps,
        validation_steps=validation_steps,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs - 1,
        callbacks=callbacks,
    )


@gin.configurable
def evalution_loop(model, loss_function, metrics, config, 
                   save_path,
                   test=None,  test_steps=None,
                   use_gpu=False, device_numbers=[0],
                   custom_callbacks=[],  
                   pretrained_weights_path=None,
                   nummodalities=2,
                  ):

    
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
        nummodalities=nummodalities
    )

    if use_gpu and torch.cuda.is_available(): 
        base_device = torch.device("cuda:{}".format(device_numbers[0]))
        model.to(base_device)
        logger.info("Sending model to {}".format(base_device))
    
    model.eval_loop(
        test,  
        epochs=0,
        test_steps=test_steps,
        callbacks=callbacks
    )


