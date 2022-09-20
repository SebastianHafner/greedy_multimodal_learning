# -*- coding: utf-8 -*-

import timeit
import gin
import sys
import numpy as np
import logging
import random
import itertools
import torch
import wandb
from pathlib import Path

from gin.config import _OPERATIVE_CONFIG

from src.utils import save_weights

logger = logging.getLogger(__name__)        


class CallbackList:
    def __init__(self, callbacks=None):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def set_model_pytoune(self, model_pytoune):
        for callback in self.callbacks:
            callback.set_model_pytoune(model_pytoune)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_forward_begin(self, batch, data):
        for callback in self.callbacks:
            callback.on_forward_begin(batch, data)

    def on_backward_end(self, batch):
        for callback in self.callbacks:
            callback.on_backward_end(batch)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_val_batch_end(self, batch, logs):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_val_batch_end(batch, logs)

    def __iter__(self):
        return iter(self.callbacks)


class Callback(object):
    def __init__(self):
        pass

    def set_config(self, config):
        self.config = config

    def set_meta_data(self, meta_data):
        self.meta_data = meta_data

    def set_save_path(self, save_path):
        self.save_path = save_path

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_model(self, model, ignore=True):
        if ignore:
            return
        self.model = model

    def set_model_pytoune(self, model_pytoune):
        self.model_pytoune = model_pytoune 

    def set_params(self, params):
        self.params = params

    def set_dataloader(self, data):
        self.data = data

    def get_dataloader(self):
        return self.data

    def get_config(self):
        return self.config

    def get_meta_data(self):
        return self.meta_data

    def get_optimizer(self):
        return self.optimizer

    def get_params(self):
        return self.params

    def get_model(self):
        return self.model

    def get_save_path(self):
        return self.save_path

    def on_epoch_begin(self, epoch, logs):
        pass

    def on_epoch_end(self, epoch, logs):
        pass

    def on_batch_begin(self, batch, logs):
        pass

    def on_batch_end(self, batch, logs):
        pass

    def on_forward_begin(self, batch, data):
        pass

    def on_backward_end(self, batch):
        pass

    def on_train_begin(self, logs):
        pass

    def on_train_end(self, logs):
        pass

    def on_val_batch_end(self, batch, logs):
        pass


@gin.configurable
class BiasMitigationStrong(Callback):
    def __init__(self, epsilon, curation_windowsize, branchnames, starting_epoch=2):
        super(BiasMitigationStrong, self).__init__()
        self.epsilon = epsilon
        self.branchnames = branchnames
        self.curation_windowsize = curation_windowsize
        self.starting_epoch = starting_epoch

        self.M_params_sar_fusion, self.M_params_opt_fusion = None, None
        self.M_params_sar_branch, self.M_params_opt_branch = None, None
        self.model_pytoune = None
        self.model_pytoune = None
        self.unlock = None
        self.d_speed, self.cls_sar, self.cls_opt = None, None, None
        self.curation_step = None

    def on_train_begin(self, logs):
        self.M_params_sar_fusion, self.M_params_opt_fusion = 0, 0
        self.M_params_sar_branch, self.M_params_opt_branch = 0, 0
        self.model_pytoune.curation_mode = False
        self.model_pytoune.caring_modality = None
        self.unlock = False

    def compute_d_speed(self):
        wn_branches, wn_fusion_modules = [0]*len(self.branchnames), [0]*len(self.branchnames)
        gn_branches, gn_fusion_modules = [0]*len(self.branchnames), [0]*len(self.branchnames)

        for name, parameter in self.model.named_parameters():
            wn = (parameter ** 2).sum().item()
            gn = (parameter.grad.data ** 2).sum().item()  # (grad ** 2).sum().item()

            if 'mmtm' in name:
                shared = True
                for ind, modal in enumerate(self.branchnames):
                    if modal in name: 
                        wn_fusion_modules[ind] += wn
                        gn_fusion_modules[ind] += gn
                        shared = False
                if shared:
                    for ind, modal in enumerate(self.branchnames):
                        wn_fusion_modules[ind] += wn
                        gn_fusion_modules[ind] += gn

            else:
                for ind, modal in enumerate(self.branchnames):
                    if modal in name: 
                        wn_branches[ind] += wn
                        gn_branches[ind] += gn

        self.M_params_sar_fusion += gn_fusion_modules[0] / wn_fusion_modules[0]
        self.M_params_opt_fusion += gn_fusion_modules[1] / wn_fusion_modules[1]
        self.M_params_sar_branch += gn_branches[0] / wn_branches[0]
        self.M_params_opt_branch += gn_branches[1] / wn_branches[1]

        cls_opt = np.log10(self.M_params_sar_fusion / self.M_params_sar_branch)
        cls_sar = np.log10(self.M_params_opt_fusion / self.M_params_opt_branch)
        d_speed = cls_opt - cls_sar

        return d_speed, cls_sar, cls_opt

    def on_batch_end(self, batch, logs):
        logs['curation_mode'] = float(self.model_pytoune.curation_mode)
        logs['caring_modality'] = self.model_pytoune.caring_modality
        logs['d_speed'] = self.d_speed
        logs['cls_sar'] = self.cls_sar
        logs['cls_opt'] = self.cls_opt

    def on_backward_end(self, batch):
        if self.unlock:
            if not self.model_pytoune.curation_mode:
                self.d_speed, self.cls_sar, self.cls_opt = self.compute_d_speed()
                if abs(self.d_speed) > self.epsilon:
                    biased_direction = np.sign(self.d_speed)
                    self.model_pytoune.curation_mode = True
                    self.curation_step = 0

                    if biased_direction == -1:  # cls opt < cls sar
                        self.model_pytoune.caring_modality = 1
                    elif biased_direction == 1:  # cls opt > cls sar
                        self.model_pytoune.caring_modality = 0 
                else:
                    self.model_pytoune.curation_mode = False 
                    self.model_pytoune.caring_modality = 0 
            else:
                self.curation_step += 1
                if self.curation_step == self.curation_windowsize:
                    self.model_pytoune.curation_mode = False
        else:
            self.d_speed, self.cls_sar, self.cls_opt = self.compute_d_speed()
            self.model_pytoune.curation_mode = False 
            self.model_pytoune.caring_modality = 0 

    def on_epoch_begin(self, epoch, logs):
        if epoch >= self.starting_epoch:
            self.unlock = True


@gin.configurable
class EarlyStopping(Callback):
    def __init__(self, *, monitor='val_f1', patience=2, verbose=True, mode='max'):
        super(EarlyStopping, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.stopped_epoch = 0
        self.counter = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            self.monitor_op = np.greater
            self.best = -np.Inf

    def on_epoch_end(self, epoch, logs):
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            self.best = current
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:

            self.stopped_epoch = epoch
            self.model_pytoune.stop_training = True

    def on_train_end(self, logs):
        if self.stopped_epoch > 0 and self.verbose:
            print('Epoch %05d: completed stopping' % (self.stopped_epoch + 1))


@gin.configurable
class ModelCheckpoint(Callback):
    def __init__(self, save_path, run_name, monitor='val_f1', verbose=1, save_best_only=True, mode='max', period=1):
        super(ModelCheckpoint, self).__init__()
        self.save_path = Path(save_path)
        self.run_name = str(run_name),
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            self.monitor_op = np.greater
            self.best = -np.Inf

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        del state['optimizer']
        return state

    def __setstate__(self, newstate):
        newstate['model'] = self.model
        newstate['optimizer'] = self.optimizer
        self.__dict__.update(newstate)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            if self.save_best_only:
                file_name = Path(self.save_path) / f'model_best_val_{self.run_name[0]}.pt'
                current = logs.get(self.monitor)
                if current is None:
                    logging.warning(f'Can save best model only with {self.monitor} available, skipping', RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print(f'Epoch {epoch}: {self.monitor} improved from {self.best:.2f} to {current:.2f}')
                            print(f'saving model to {file_name}')
                        self.best = current
                        save_weights(self.model, self.optimizer, file_name)
                    else:
                        if self.verbose > 0:
                            print(f'Epoch {epoch}: {self.monitor} did not improve')
            else:
                file_name = Path(self.save_path) / f'model_epoch{epoch}_{self.run_name[0]}.pt'
                if self.verbose > 0:
                    print(f'Epoch {epoch}: saving model to {file_name}')
                save_weights(self.model, self.optimizer, file_name)




@gin.configurable
class WBLoggingCallback(Callback):
    def __init__(self, on: bool, run_name: str, project: str, log_frequency: int, metrics, other_metrics: list = None):

        wandb.init(
            name=run_name,
            entity='spacenet7',
            project=project,
            tags=['run', 'urban', 'extraction', 'segmentation', ],
            mode='online' if on else 'disabled',
        )

        self.train_logfreq = log_frequency

        # adding unimodal metrics
        metrics = [f'{metric}{suffix}' for metric in metrics for suffix in ['', '_sar', '_opt']]

        self.train_metrics = list(metrics)
        self.train_metrics.append('loss')
        self.train_values = [[] for _ in range(len(self.train_metrics))]
        if other_metrics is not None:
            for me in other_metrics:
                self.train_metrics.append(me)
                self.train_values.append([])

        self.eval_metrics = [f'{split}{metric}' for metric in (metrics + ['loss']) for split in ['', 'val_', 'test_']]

        self.epoch = 0
        self.epochs = None
        self.steps = None

    def on_train_begin(self, logs):
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']

    def on_epoch_begin(self, epoch, logs):
        self.epoch = epoch
        self._reset_train_lists()

    def on_epoch_end(self, epoch, logs):
        log = {'epoch': epoch}
        for metric, values in logs.items():
            if metric in self.eval_metrics:
                log[metric] = values
        wandb.log(log)

    def on_batch_end(self, batch, logs):
        if not batch % (self.train_logfreq + 1) == 0:
            for i, metric in enumerate(self.train_metrics):
                if metric in logs.keys():
                    self.train_values[i].append(logs[metric])
        else:
            log_dict = {
                'step': (self.epoch - 1) * self.steps + batch,
                'epoch_float': self.epoch - 1 + batch / self.steps,
            }

            for metric, values in zip(self.train_metrics, self.train_values):
                if metric == 'curation_mode':
                    log_dict['regular_step_rate'] = np.sum(np.logical_not(values)) / self.train_logfreq * 100
                    log_dict['rebalancing_step_rate'] = np.sum(values) / self.train_logfreq * 100
                if metric == 'caring_modality':
                    index = self.train_metrics.index('curation_mode')
                    curation_mode = np.array(self.train_values[index], dtype=bool)
                    actual_rebalancing_steps = np.array(values)[curation_mode]
                    log_dict['rebalancing_step_rate_sar'] = np.sum(np.logical_not(actual_rebalancing_steps)) /\
                                                            self.train_logfreq * 100
                    log_dict['rebalancing_step_rate_opt'] = np.sum(actual_rebalancing_steps) / self.train_logfreq * 100
                else:
                    if values:
                        log_dict[f'train_{metric}'] = np.mean(values)
            wandb.log(log_dict)
            self._reset_train_lists()

    def _reset_train_lists(self):
        for i in range(len(self.train_values)):
            self.train_values[i] = []


@gin.configurable
class ProgressionCallback(Callback):
    def __init__(self, other_metrics: list = None):
         
        self.other_metrics = []
        if other_metrics is not None:
            for me in other_metrics:
                self.other_metrics.append(me)

    def on_train_begin(self, logs):
        self.metrics = ['loss'] + self.model_pytoune.metrics_names
        self.epochs = self.params['epochs']
        self.steps = self.params['steps']

    def on_epoch_begin(self, epoch, logs):
        self.step_times_sum = 0.
        self.epoch = epoch
        sys.stdout.write(f'Epoch {self.epoch}/{self.epochs}')
        sys.stdout.flush()

    def on_epoch_end(self, epoch, logs):
        epoch_total_time = logs['time']

        metrics_str = self._get_metrics_string(logs)
        iol_str = self._get_iol_string(logs)
        if self.steps is not None:

            epoch_time = timeit.default_timer() - logs['epoch_begin_time']
            # print(f'Epoch {self.epoch}/{self.epochs} {epoch_total_time:.2fs}/{epoch_time:.2fs}: Step {self.steps}/{self.steps}: {metrics_str}. {iol_str}')

            print("\rEpoch %d/%d %.2fs/%.2fs: Step %d/%d: %s. %s" %
                  (self.epoch, self.epochs, epoch_total_time, epoch_time, self.steps, self.steps, metrics_str, iol_str))

        else:
            print("\rEpoch %d/%d %.2fs/%.2fs: Step %d/%d: %s. %s" %
                  (self.epoch, self.epochs, epoch_total_time, timeit.default_timer()-logs['epoch_begin_time'], self.last_step, self.last_step, metrics_str, iol_str))

    def on_batch_end(self, batch, logs):
        self.step_times_sum += timeit.default_timer() - logs['batch_begin_time']

        metrics_str = self._get_metrics_string(logs)
        iol_str = self._get_iol_string(logs)

        times_mean = self.step_times_sum / batch
        if self.steps is not None:
            eta = times_mean * (self.steps - batch)
            _str = f'Epoch {self.epoch}/{self.epochs} ETA {eta:.2f} Step {batch}/{self.steps}: {metrics_str} {iol_str}'
            sys.stdout.write("\r%s" % _str)
            if 'cumsum_iol' in iol_str:
                sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            sys.stdout.write("\rEpoch %d/%d %.2fs/step Step %d: %s. %s" %
                             (self.epoch, self.epochs, times_mean, batch, metrics_str, iol_str))
            sys.stdout.flush()
            self.last_step = batch

    def _get_metrics_string(self, logs):
        train_metrics_str_gen = ('{}: {:f}'.format(k, logs[k]) for k in self.metrics if logs.get(k) is not None)
        val_metrics_str_gen = ('{}: {:f}'.format('val_' + k, logs['val_' + k]) for k in self.metrics
                               if logs.get('val_' + k) is not None)
        return ', '.join(itertools.chain(train_metrics_str_gen, val_metrics_str_gen))

    def _get_iol_string(self, logs):
        str_gen = ['{}: {:f}'.format(k, logs[k]) for k in self.other_metrics if logs.get(k) is not None]
        return  ', '.join(str_gen)


class EvalProgressionCallback(Callback):
    def __init__(self, phase, metrics_names, steps=None):
        self.params = {}
        self.params['steps'] = steps
        self.params['phase'] = phase 
        self.metrics = metrics_names

        super(EvalProgressionCallback, self).__init__()

    def _get_metrics_string(self, logs):
        metrics_str_gen = ('{}: {:f}'.format(self.params['phase'] + '_' + k, logs[k]) for k in self.metrics
                               if logs.get(k) is not None)
        return ', '.join(metrics_str_gen)

    def on_batch_begin(self, batch, logs):
        if batch == 1:
            self.step_times_sum = 0.
        
        self.steps = self.params['steps']

    def on_batch_end(self, batch, logs):
        self.step_times_sum += timeit.default_timer()-logs['batch_begin_time']

        metrics_str = self._get_metrics_string(logs)
        times_mean = self.step_times_sum / batch
        if self.steps is not None:
            remaining_time = times_mean * (self.steps - batch)
            
            sys.stdout.write("\r%s ETA %.2fs Step %d/%d: %s." %
                             (self.params['phase'], remaining_time, batch, self.steps, metrics_str))
            sys.stdout.flush()
        else:
            sys.stdout.write("\r%s %.2fs/step Step %d: %s." %
                             (self.params['phase'], times_mean, batch, metrics_str))
            sys.stdout.flush()
            self.last_step = batch


class LambdaCallback(Callback):
    def __init__(self,
                 on_epoch_begin=None,
                 on_epoch_end=None,
                 on_batch_begin=None,
                 on_batch_end=None,
                 on_train_begin=None,
                 on_train_end=None):
        super(LambdaCallback, self).__init__()
        if on_epoch_begin is not None:
            self.on_epoch_begin = on_epoch_begin
        else:
            self.on_epoch_begin = lambda epoch, logs: None
        if on_epoch_end is not None:
            self.on_epoch_end = on_epoch_end
        else:
            self.on_epoch_end = lambda epoch, logs: None
        if on_batch_begin is not None:
            self.on_batch_begin = on_batch_begin
        else:
            self.on_batch_begin = lambda batch, logs: None
        if on_batch_end is not None:
            self.on_batch_end = on_batch_end
        else:
            self.on_batch_end = lambda batch, logs: None
        if on_train_begin is not None:
            self.on_train_begin = on_train_begin
        else:
            self.on_train_begin = lambda logs: None
        if on_train_end is not None:
            self.on_train_end = on_train_end
        else:
            self.on_train_end = lambda logs: None