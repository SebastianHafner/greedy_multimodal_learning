import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import glob
import gin
from gin.config import _CONFIG

from src.balanced_mmtm import MMTM_mitigate as MMTM
from src.balanced_mmtm import get_rescale_weights


@gin.configurable
class MMTM_DSUNet(nn.Module):
    def __init__(self,
                 mmtm_off=False,
                 mmtm_rescale_eval_file_path=None,
                 mmtm_rescale_training_file_path=None,
                 device='cuda:0',
                 saving_mmtm_scales=False,
                 saving_mmtm_squeeze_array=False,
                 ):
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
