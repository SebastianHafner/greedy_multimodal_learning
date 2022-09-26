import numpy as np
import timeit
import torch
import math

from src.callbacks import EvalProgressionCallback

import logging
logger = logging.getLogger(__name__)


class Framework:
    def __init__(self, model, optimizer, loss_function, nummodalities, *, metrics=[], verbose=True, config, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.modality_names = ['sar', 'opt']

        self.metrics = metrics
        self.metrics_names = [metric.__name__ for metric in self.metrics]
        self.modalitywise_metrics_names = [f'{x}_{m}' for x in self.metrics_names for m in self.modality_names]
        self.all_metrics_names = list(self.metrics_names) +\
                                 [f'{x}_{m}' for x in list(self.metrics_names) for m in self.modality_names] + ['loss']

        self.verbose = verbose
        self.verbose_logs = {} 
        self.nummodalities = nummodalities

        self.config = config
        self.device = device
        self.stop_training = None

        # balanced multi-modal learning
        self.curation_mode = False
        self.caring_modality = None

        self.defaultfields = ['indices', 'loss', 'metrics', 'viewwises_metrics', 'number', 'size']

        # iterator values training
        self.losses_sum = 0.
        self.metrics_sum = np.zeros(len(self.metrics))
        self.metrics_permodal_sum = np.zeros((nummodalities, len(self.metrics)))
        self.sizes_sum = 0.
        self.extra_lists = {}
        self.indices_list = []

        # iterator values evaluation
        self.eval_losses_sum = 0.
        self.eval_metrics_sum = np.zeros(len(self.metrics))
        self.eval_metrics_permodal_sum = np.zeros((nummodalities, len(self.metrics)))
        self.eval_sizes_sum = 0.
        self.eval_extra_lists = {}
        self.eval_indices_list = []

    def train_loop(self, train_generator, test_generator=None, valid_generator=None, *, epochs=1000,
                   callback_list=None):

        validation_steps = None if valid_generator is None else len(valid_generator)
        test_steps = None if test_generator is None else len(test_generator)

        self.stop_training = False

        callback_list.on_train_begin({})
        for epoch in range(1, epochs + 1):
            self._reset_train_variables()
            callback_list.on_epoch_begin(epoch, {})
            epoch_begin_time = timeit.default_timer()

            self.model.train(True)
            for step_index, (indices, x, y) in enumerate(train_generator):

                batch_begin_time = timeit.default_timer()
                batch_ind = step_index + 1

                callback_list.on_batch_begin(batch_ind, {})
                callback_list.on_forward_begin(batch_ind, (indices, x, y))

                step = {'number': batch_ind, 'indices': indices, 'size': self._get_batch_size(x, y)}

                x = [tensor.to(self.device) for tensor in x]
                y = y.to(self.device)

                self.optimizer.zero_grad()
                pred_y_eval, pred_y = self.model(*x, curation_mode=self.curation_mode,
                                                 caring_modality=self.caring_modality)
                loss_tensor = self.loss_function(pred_y, y)

                with torch.no_grad():
                    step['metrics'] = self._compute_metrics(y, pred_y_eval)
                    step['modalitywise_metrics'] = self._compute_metrics_multiple_inputs(y, pred_y)

                loss_tensor.backward()
                callback_list.on_backward_end(step['number'])

                self.optimizer.step()

                loss = loss_tensor.item()
                step['loss'] = loss
                if math.isnan(step['loss']):
                    self.stop_training = True

                self.losses_sum += step['loss'] * step['size']
                self.metrics_sum += step['metrics'] * step['size']
                self.metrics_permodal_sum += step['modalitywise_metrics'] * step['size']
                self.sizes_sum += step['size']
                self.indices_list.append(indices)

                metrics_dict = dict(zip(self.metrics_names, step['metrics']))
                for i in range(self.nummodalities):
                    names = [f'{x}_{"sar" if i == 0 else "opt"}' for x in self.metrics_names]
                    metrics_dict.update(dict(zip(names, step['modalitywise_metrics'][i])))

                for key, value in step.items():
                    if key not in self.defaultfields:
                        if key in self.extra_lists:
                            self.extra_lists[key].append(value)
                        else:
                            self.extra_lists[key] = [value]

                batch_total_time = timeit.default_timer() - batch_begin_time
                batch_logs = {'batch': batch_ind, 'size': step['size'], 'time': batch_total_time,
                              'batch_begin_time': batch_begin_time, 'loss': step['loss'], **metrics_dict}

                callback_list.on_batch_end(batch_ind, batch_logs)

            train_dict = {
                'loss': self._get_loss(self.sizes_sum, self.losses_sum),
                'train_indices': self._get_indices(self.sizes_sum, self.indices_list),
                **{f'train_{k}': v for k, v in self.extra_lists.items()},
                **self._get_metrics(self.sizes_sum, self.metrics_sum, self.metrics_permodal_sum)
            }

            # evaluation on validation and test set
            val_dict = self._training_model_evaluation(valid_generator, 'val', steps=validation_steps)
            test_dict = self._training_model_evaluation(test_generator, 'test', steps=test_steps)

            epoch_log = {
                'epoch': epoch,
                'time': timeit.default_timer() - epoch_begin_time,
                'epoch_begin_time': epoch_begin_time,
                **train_dict, **val_dict, **test_dict
            }

            callback_list.on_epoch_end(epoch, epoch_log)

            if self.stop_training:
                break

        callback_list.on_train_end({})

    def _training_model_evaluation(self, eval_generator, phase, *, steps=None) -> dict:
        self._reset_eval_variables()

        if steps is None:
            steps = len(eval_generator)

        eval_callback = EvalProgressionCallback(phase=phase, steps=steps, metrics_names=self.all_metrics_names)

        self.model.eval()
        with torch.no_grad():
            for step_index, (indices, x, y) in enumerate(eval_generator):

                batch_begin_time = timeit.default_timer()
                batch_ind = step_index + 1

                eval_callback.on_batch_begin(batch_ind, {})

                step = {
                    'number': batch_ind,
                    'indices': indices
                }

                batch_size = self._get_batch_size(x, y)
                step['size'] = batch_size

                x = [tensor.to(self.device) for tensor in x]
                y = y.to(self.device)

                pred_y_eval, pred_y, *_ = self.model(*x)
                loss_tensor = self.loss_function(pred_y, y)

                with torch.no_grad():
                    step['metrics'] = self._compute_metrics(y, pred_y_eval)
                    step['modalitywise_metrics'] = self._compute_metrics_multiple_inputs(y, pred_y)

                step['loss'] = float(loss_tensor.item())

                self.eval_losses_sum += step['loss'] * batch_size
                self.eval_metrics_sum += step['metrics'] * batch_size
                self.eval_metrics_permodal_sum += step['modalitywise_metrics'] * batch_size
                self.eval_sizes_sum += step['size']
                self.eval_indices_list.append(indices)

                metrics_dict = dict(zip(self.metrics_names, step['metrics']))

                for i in range(self.nummodalities):
                    names = [f'{x}_{"sar" if i == 0 else "opt"}' for x in self.metrics_names]
                    metrics_dict.update(dict(zip(names, step['modalitywise_metrics'][i])))

                batch_total_time = timeit.default_timer() - batch_begin_time

                batch_logs = {'batch': batch_ind, 'size': step['size'], 'time': batch_total_time,
                              'batch_begin_time': batch_begin_time, 'loss': step['loss'], **metrics_dict}
                eval_callback.on_batch_end(batch_ind, batch_logs)

        info_dict = {
            f'{phase}_loss': self._get_loss(self.eval_sizes_sum, self.eval_losses_sum),
            f'{phase}_indices': self._get_indices(self.eval_sizes_sum, self.eval_indices_list),
            **self._get_metrics(self.eval_sizes_sum, self.eval_metrics_sum, self.eval_metrics_permodal_sum, phase)
        }

        return info_dict

    @staticmethod
    def _get_loss(sizes_sum, losses_sum):
        if sizes_sum == 0:
            return 0
        else:
            return losses_sum / sizes_sum

    def _get_metrics(self, sizes_sum, metrics_sum, metrics_permodal_sum, phase=None):
        if sizes_sum == 0:
            metrics_dict = dict(zip([f'{phase}_{m}' for m in self.metrics_names], np.zeros(len(self.metrics_names))))
        else:
            metrics_dict = dict(zip([f'{phase}_{m}' for m in self.metrics_names], metrics_sum / sizes_sum))
            for i in range(self.nummodalities):
                names = [f'{phase}_{x}_{"sar" if i == 0 else "opt"}' for x in self.metrics_names]
                metrics_dict.update(dict(zip(names, metrics_permodal_sum[i] / sizes_sum)))
        return metrics_dict

    @staticmethod
    def _get_indices(sizes_sum, indices_list):
        if sizes_sum == 0:
            return []
        elif indices_list[0] is None:
            return []
        else:
            return np.concatenate(indices_list, axis=0)

    def _reset_eval_variables(self):
        self.eval_losses_sum = 0.
        self.eval_metrics_sum = np.zeros(len(self.metrics))
        self.eval_metrics_permodal_sum = np.zeros((self.nummodalities, len(self.metrics)))
        self.eval_sizes_sum = 0.
        self.eval_extra_lists = {}
        self.eval_indices_list = []

    def _reset_train_variables(self):
        self.losses_sum = 0.
        self.metrics_sum = np.zeros(len(self.metrics))
        self.metrics_permodal_sum = np.zeros((self.nummodalities, len(self.metrics)))
        self.sizes_sum = 0.
        self.extra_lists = {}
        self.indices_list = []

    def _compute_metrics(self, y, pred_y):
        return np.array([float(metric(y, pred_y)) for metric in self.metrics])

    def _compute_metrics_multiple_inputs(self, y, list_pred_y):
        return np.array([self._compute_metrics(y, pred_y) for pred_y in list_pred_y])

    def _get_batch_size(self, x, y):
        if torch.is_tensor(x) or isinstance(x, np.ndarray):
            return len(x)
        if torch.is_tensor(y) or isinstance(y, np.ndarray):
            return len(y)
        return 1
