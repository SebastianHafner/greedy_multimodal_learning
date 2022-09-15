# -*- coding: utf-8 -*-

import logging
import torch
import gin
from src.framework import Model_

logger = logging.getLogger(__name__)


def _load_pretrained_model(model, save_path):
    checkpoint = torch.load(save_path)
    model_dict = model.state_dict()
    model_dict.update(checkpoint['model']) 
    model.load_state_dict(model_dict, strict=False)
    logger.info("Done reloading!")


@gin.configurable
def training_loop(model, loss_function, metrics, optimizer, config, save_path,  steps_per_epoch, train=None,
                  valid=None, test=None, test_steps=None, validation_steps=None, custom_callbacks=[],
                  n_epochs=100, verbose=True, nummodalities=2):

    callbacks = list(custom_callbacks)
    
    # Configure callbacks
    for clbk in callbacks:
        clbk.set_save_path(save_path)
        clbk.set_model(model, ignore=False)
        clbk.set_optimizer(optimizer)
        clbk.set_config(config)

    model = Model_(
        model=model,
        optimizer=optimizer, 
        loss_function=loss_function, 
        metrics=metrics,
        verbose=verbose,
        nummodalities=nummodalities,
        config=config,
    )
            
    for clbk in callbacks:
        clbk.set_model_pytoune(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Sending model to {device}")
        
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Sending model to {device}")
    
    model.eval_loop(
        test,  
        epochs=0,
        test_steps=test_steps,
        callbacks=callbacks
    )


