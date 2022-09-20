import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import gin
from gin.config import _CONFIG
import pickle
from src.utils import numpy_to_torch, torch_to


@gin.configurable
class MMTM_DSUNet(nn.Module):
    def __init__(self, mmtm_off=False, mmtm_rescale_eval_file_path=None, mmtm_rescale_training_file_path=None,
                 device='cuda', saving_mmtm_scales=False, saving_mmtm_squeeze_array=False):
        super(MMTM_DSUNet, self).__init__()

        self.mmtm_off = mmtm_off
        if self.mmtm_off:
            self.mmtm_rescale = get_rescale_weights(
                mmtm_rescale_eval_file_path,
                mmtm_rescale_training_file_path,
                validation=False,
                starting_mmtmindice=1,
                mmtmpositions=4,
                device=torch.device(device),
            )

        self.saving_mmtm_scales = saving_mmtm_scales
        self.saving_mmtm_squeeze_array = saving_mmtm_squeeze_array

        self.inc_sar = InConv(2, 64, DoubleConv)
        self.inc_opt = InConv(4, 64, DoubleConv)

        self.mmtm1 = self.mmtm1 = MMTM(64, 64, 1)

        self.max1_sar = nn.MaxPool2d(2)
        self.max1_opt = nn.MaxPool2d(2)

        self.conv1_sar = DoubleConv(64, 128)
        self.conv1_opt = DoubleConv(64, 128)

        self.mmtm2 = MMTM(128, 128, 1)

        self.max2_sar = nn.MaxPool2d(2)
        self.max2_opt = nn.MaxPool2d(2)

        self.conv2_sar = DoubleConv(128, 128)
        self.conv2_opt = DoubleConv(128, 128)

        self.mmtm3 = MMTM(128, 128, 1)

        self.up1_sar = nn.ConvTranspose2d(128, 128, (2, 2), stride=(2, 2))
        self.up1_opt = nn.ConvTranspose2d(128, 128, (2, 2), stride=(2, 2))

        self.conv3_sar = DoubleConv(256, 64)
        self.conv3_opt = DoubleConv(256, 64)

        self.mmtm4 = MMTM(64, 64, 1)

        self.up2_sar = nn.ConvTranspose2d(64, 64, (2, 2), stride=(2, 2))
        self.up2_opt = nn.ConvTranspose2d(64, 64, (2, 2), stride=(2, 2))

        self.conv4_sar = DoubleConv(128, 64)
        self.conv4_opt = DoubleConv(128, 64)

        self.outc_sar = OutConv(64, 1)
        self.outc_opt = OutConv(64, 1)

    def forward(self, x_sar, x_opt, curation_mode=False, caring_modality=None):

        mmtm_kwargs = {
            'return_scale': self.saving_mmtm_scales,
            'return_squeezed_mps': self.saving_mmtm_squeeze_array,
            'turnoff_cross_modal_flow': True if self.mmtm_off else False,
            'curation_mode': curation_mode,
            'caring_modality': caring_modality
        }

        scales = []
        squeezed_mps = []

        features_sar = self.inc_sar(x_sar)
        features_opt = self.inc_opt(x_opt)

        mmtm_kwargs['average_squeezemaps'] = self.mmtm_rescale[1] if self.mmtm_off else None
        features_sar, features_opt, scale, squeezed_mp = self.mmtm1(features_sar, features_opt, **mmtm_kwargs)
        scales.append(scale)
        squeezed_mps.append(squeezed_mp)
        skip1_sar, skip1_opt = features_sar, features_opt

        features_sar = self.max1_sar(features_sar)
        features_sar = self.conv1_sar(features_sar)
        features_opt = self.max1_opt(features_opt)
        features_opt = self.conv1_opt(features_opt)

        mmtm_kwargs['average_squeezemaps'] = self.mmtm_rescale[2] if self.mmtm_off else None
        features_sar, features_opt, scale, squeezed_mp = self.mmtm2(features_sar, features_opt, **mmtm_kwargs)
        skip2_sar, skip2_opt = features_sar, features_opt
        scales.append(scale)
        squeezed_mps.append(squeezed_mp)

        features_sar = self.max2_sar(features_sar)
        features_sar = self.conv2_sar(features_sar)
        features_opt = self.max2_opt(features_opt)
        features_opt = self.conv2_opt(features_opt)

        mmtm_kwargs['average_squeezemaps'] = self.mmtm_rescale[3] if self.mmtm_off else None
        features_sar, features_opt, scale, squeezed_mp = self.mmtm3(features_sar, features_opt, **mmtm_kwargs)
        scales.append(scale)
        squeezed_mps.append(squeezed_mp)

        features_sar = self.up1_sar(features_sar)
        diffY = skip2_sar.detach().size()[2] - features_sar.detach().size()[2]
        diffX = skip2_sar.detach().size()[3] - features_sar.detach().size()[3]
        features_sar = F.pad(features_sar, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        features_sar = torch.cat([skip2_sar, features_sar], dim=1)
        features_sar = self.conv3_sar(features_sar)

        features_opt = self.up1_opt(features_opt)
        features_opt = F.pad(features_opt, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        features_opt = torch.cat([skip2_opt, features_opt], dim=1)
        features_opt = self.conv3_opt(features_opt)

        mmtm_kwargs['average_squeezemaps'] = self.mmtm_rescale[4] if self.mmtm_off else None
        features_sar, features_opt, scale, squeezed_mp = self.mmtm4(features_sar, features_opt, **mmtm_kwargs)
        scales.append(scale)
        squeezed_mps.append(squeezed_mp)

        features_sar = self.up2_sar(features_sar)
        diffY = skip1_sar.detach().size()[2] - features_sar.detach().size()[2]
        diffX = skip1_sar.detach().size()[3] - features_sar.detach().size()[3]
        features_sar = F.pad(features_sar, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        features_sar = torch.cat([skip1_sar, features_sar], dim=1)
        features_sar = self.conv4_sar(features_sar)

        features_opt = self.up2_opt(features_opt)
        features_opt = F.pad(features_opt, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        features_opt = torch.cat([skip1_opt, features_opt], dim=1)
        features_opt = self.conv4_opt(features_opt)

        out_sar = torch.sigmoid(self.outc_sar(features_sar))
        out_opt = torch.sigmoid(self.outc_opt(features_opt))

        return (out_sar + out_opt) / 2, [out_sar, out_opt], scales, squeezed_mps


@gin.configurable
class MMTM(nn.Module):
    def __init__(self, dim_sar, dim_opt, ratio, device='cuda'):
        super(MMTM, self).__init__()
        dim = dim_sar + dim_opt
        dim_out = int(2 * dim / ratio)

        self.running_avg_weight_sar = torch.zeros(dim_sar).to(device)
        self.running_avg_weight_opt = torch.zeros(dim_opt).to(device)
        self.step = 0

        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_sar = nn.Linear(dim_out, dim_sar)
        self.fc_opt = nn.Linear(dim_out, dim_opt)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, sar, opt, return_scale=False, return_squeezed_mps=False, turnoff_cross_modal_flow=False,
                average_squeezemaps=None, curation_mode=False, caring_modality=0):

        if not turnoff_cross_modal_flow:
            squeeze_array = []
            for tensor in [sar, opt]:
                tview = tensor.view(tensor.shape[:2] + (-1,))
                squeeze_array.append(torch.mean(tview, dim=-1))

            squeeze = torch.cat(squeeze_array, 1)
            excitation = self.fc_squeeze(squeeze)
            excitation = self.relu(excitation)

            sar_out = self.fc_sar(excitation)
            opt_out = self.fc_opt(excitation)

        else:
            tview = sar.view(sar.shape[:2] + (-1,))
            squeeze = torch.cat([torch.mean(tview, dim=-1),
                                 torch.stack(sar.shape[0] * [average_squeezemaps[1]])], 1)
            excitation = self.relu(self.fc_squeeze(squeeze))

            sar_out = self.fc_sar(excitation)

            tview = opt.view(opt.shape[:2] + (-1,))
            squeeze = torch.cat([torch.stack(opt.shape[0] * [average_squeezemaps[0]]),
                                 torch.mean(tview, dim=-1)], 1)
            excitation = self.relu(self.fc_squeeze(squeeze))

            opt_out = self.fc_opt(excitation)

        sar_out = self.sigmoid(sar_out)
        opt_out = self.sigmoid(opt_out)

        self.running_avg_weight_sar = (sar_out.mean(0) + self.running_avg_weight_sar * self.step).detach() / (
                self.step + 1)
        self.running_avg_weight_opt = (opt_out.mean(0) + self.running_avg_weight_opt * self.step).detach() / (
                self.step + 1)

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
                sar_out = torch.stack(sar_out.shape[0] * [
                    self.running_avg_weight_sar
                ]).view(sar_out.shape + (1,) * dim_diff)

            elif caring_modality == 1:
                dim_diff = len(sar.shape) - len(sar_out.shape)
                sar_out = sar_out.view(sar_out.shape + (1,) * dim_diff)

                dim_diff = len(opt.shape) - len(opt_out.shape)
                opt_out = torch.stack(opt_out.shape[0] * [
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
            if len(data) < mmtmid + 1:
                data.append({})
            for i, viewdd in enumerate(batch[mmtmid]):
                data[mmtmid].setdefault('view_%d' % i, []).append(np.array(viewdd))

    for mmtmid in range(len(data)):
        for k, v in data[mmtmid].items():
            data[mmtmid][k] = np.concatenate(data[mmtmid][k])[np.argsort(his_epo['test_indices'][0])]

    return data


def get_rescale_weights(eval_save_path,
                        training_save_path,
                        key='test_squeezedmaps_array_list',
                        validation=False,
                        starting_mmtmindice=1,
                        mmtmpositions=4,
                        device=None,
                        ):
    data = get_mmtm_outputs(eval_save_path, mmtmpositions - starting_mmtmindice, key)

    with open(os.path.join(training_save_path, 'history.pickle'), 'rb') as f:
        his_ori = pickle.load(f)

    selected_indices = his_ori['val_indices'][0] if validation else his_ori['train_indices'][0]

    mmtm_weights = []
    for mmtmid in range(mmtmpositions):
        if mmtmid < starting_mmtmindice:
            mmtm_weights.append(None)
        else:
            weights = [data[mmtmid - starting_mmtmindice][k][selected_indices].mean(0) \
                       for k in sorted(data[mmtmid - starting_mmtmindice].keys())]
            if device is not None:
                weights = numpy_to_torch(weights)
                weights = torch_to(weights, device)
            mmtm_weights.append(weights)

    return mmtm_weights


# sub-parts of the U-Net model
class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_block):
        super(InConv, self).__init__()
        self.conv = conv_block(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


