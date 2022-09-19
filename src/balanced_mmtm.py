import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import glob
import gin
from gin.config import _CONFIG

from src.utils import numpy_to_torch, torch_to

@gin.configurable
class MMTM_mitigate(nn.Module):
    def __init__(self, dim_sar, dim_opt, ratio, device='cuda', SEonly=False, shareweight=False):
        super(MMTM_mitigate, self).__init__()
        dim = dim_sar + dim_opt
        dim_out = int(2*dim/ratio)
        self.SEonly = SEonly
        self.shareweight = shareweight

        self.running_avg_weight_sar = torch.zeros(dim_sar).to(device)
        self.running_avg_weight_opt = torch.zeros(dim_opt).to(device)
        self.step = 0

        if self.SEonly:
            self.fc_squeeze_sar = nn.Linear(dim_sar, dim_out)
            self.fc_squeeze_opt = nn.Linear(dim_opt, dim_out)
        else:    
            self.fc_squeeze = nn.Linear(dim, dim_out)

        if self.shareweight:
            assert dim_sar == dim_opt
            self.fc_excite = nn.Linear(dim_out, dim_sar)
        else:
            self.fc_sar = nn.Linear(dim_out, dim_sar)
            self.fc_opt = nn.Linear(dim_out, dim_opt)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, sar, opt, return_scale=False, return_squeezed_mps=False, turnoff_cross_modal_flow=False,
                average_squeezemaps=None, curation_mode=False, caring_modality=0):

        if self.SEonly:
            tview = sar.view(sar.shape[:2] + (-1,))
            squeeze = torch.mean(tview, dim=-1)
            excitation = self.fc_squeeze_sar(squeeze)
            sar_out = self.fc_sar(self.relu(excitation))

            tview = opt.view(opt.shape[:2] + (-1,))
            squeeze = torch.mean(tview, dim=-1)
            excitation = self.fc_squeeze_opt(squeeze)
            opt_out = self.fc_opt(self.relu(excitation))

        else:
            if turnoff_cross_modal_flow:
                tview = sar.view(sar.shape[:2] + (-1,))
                squeeze = torch.cat([torch.mean(tview, dim=-1), 
                    torch.stack(sar.shape[0]*[average_squeezemaps[1]])], 1)
                excitation = self.relu(self.fc_squeeze(squeeze))

                if self.shareweight:
                    sar_out = self.fc_excite(excitation)
                else:
                    sar_out = self.fc_sar(excitation)

                tview = opt.view(opt.shape[:2] + (-1,))
                squeeze = torch.cat([torch.stack(opt.shape[0]*[average_squeezemaps[0]]),
                    torch.mean(tview, dim=-1)], 1)
                excitation = self.relu(self.fc_squeeze(squeeze))
                if self.shareweight:
                    opt_out = self.fc_excite(excitation)
                else:
                    opt_out = self.fc_opt(excitation)

            else: 
                squeeze_array = []
                for tensor in [sar, opt]:
                    tview = tensor.view(tensor.shape[:2] + (-1,))
                    squeeze_array.append(torch.mean(tview, dim=-1))

                squeeze = torch.cat(squeeze_array, 1)
                excitation = self.fc_squeeze(squeeze)
                excitation = self.relu(excitation)

                if self.shareweight:
                    sar_out = self.fc_excite(excitation)
                    opt_out = self.fc_excite(excitation)
                else:
                    sar_out = self.fc_sar(excitation)
                    opt_out = self.fc_opt(excitation)

        sar_out = self.sigmoid(sar_out)
        opt_out = self.sigmoid(opt_out)

        self.running_avg_weight_sar = (sar_out.mean(0) + self.running_avg_weight_sar*self.step).detach()/(self.step+1)
        self.running_avg_weight_opt = (opt_out.mean(0) + self.running_avg_weight_opt*self.step).detach()/(self.step+1)
        
        self.step += 1

        if return_scale:
            scales = [sar_out.cpu(), opt_out.cpu()]
        else:
            scales = None

        if return_squeezed_mps:
            squeeze_array = [x.cpu() for x in squeeze_array]
        else:
            squeeze_array = None

        if not curation_mode:
            dim_diff = len(sar.shape) - len(sar_out.shape)
            sar_out = sar_out.view(sar_out.shape + (1,) * dim_diff)

            dim_diff = len(opt.shape) - len(opt_out.shape)
            opt_out = opt_out.view(opt_out.shape + (1,) * dim_diff)
        
        else:
            if caring_modality == 0:
                dim_diff = len(opt.shape) - len(opt_out.shape)
                opt_out = opt_out.view(opt_out.shape + (1,) * dim_diff)

                dim_diff = len(sar.shape) - len(sar_out.shape)
                sar_out = torch.stack(sar_out.shape[0]*[
                        self.running_avg_weight_sar
                    ]).view(sar_out.shape + (1,) * dim_diff)
                
            elif caring_modality == 1:
                dim_diff = len(sar.shape) - len(sar_out.shape)
                sar_out = sar_out.view(sar_out.shape + (1,) * dim_diff)

                dim_diff = len(opt.shape) - len(opt_out.shape)
                opt_out = torch.stack(opt_out.shape[0]*[
                        self.running_avg_weight_opt
                    ]).view(opt_out.shape + (1,) * dim_diff)

        return sar * sar_out, opt * opt_out, scales, squeeze_array


def get_mmtm_outputs(eval_save_path, mmtm_recorded, key):
    with open(os.path.join(eval_save_path, 'history.pickle'), 'rb') as f:
        his_epo = pickle.load(f)
    
    print(his_epo.keys())
    data = []
    for batch in his_epo[key][0]:
        assert mmtm_recorded == len(batch)

        for mmtmid in range(len(batch)):
            if len(data)<mmtmid+1:
                data.append({})
            for i, viewdd in enumerate(batch[mmtmid]):
                data[mmtmid].setdefault('view_%d'%i, []).append(np.array(viewdd))
           
    for mmtmid in range(len(data)):
        for k, v in data[mmtmid].items():
            data[mmtmid][k] = np.concatenate(data[mmtmid][k])[np.argsort(his_epo['test_indices'][0])]  

    return data


def get_rescale_weights(eval_save_path, 
                        training_save_path, 
                        key='test_squeezedmaps_array_list',
                        validation=False, 
                        starting_mmtmindice = 1, 
                        mmtmpositions=4,
                        device=None,
                        ):
    data = get_mmtm_outputs(eval_save_path, mmtmpositions-starting_mmtmindice, key)
    
    with open(os.path.join(training_save_path, 'history.pickle'), 'rb') as f:
        his_ori = pickle.load(f)

    selected_indices = his_ori['val_indices'][0] if validation else his_ori['train_indices'][0] 
      
    mmtm_weights = []        
    for mmtmid in range(mmtmpositions):
        if mmtmid < starting_mmtmindice:
            mmtm_weights.append(None)
        else:
            weights = [data[mmtmid-starting_mmtmindice][k][selected_indices].mean(0) \
                        for k in sorted(data[mmtmid-starting_mmtmindice].keys())]
            if device is not None:
                weights = numpy_to_torch(weights)
                weights = torch_to(weights, device)
            mmtm_weights.append(weights)
        
    return mmtm_weights




