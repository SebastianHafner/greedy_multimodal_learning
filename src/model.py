import torch
import torch.nn as nn
import torch.nn.functional as F
import gin


@gin.configurable
class MMTM_DSUNet(nn.Module):
    def __init__(self, mmtm_off=False, device='cuda'):
        super(MMTM_DSUNet, self).__init__()

        self.mmtm_off = mmtm_off

        self.inc_sar = InConv(2, 64, DoubleConv)
        self.inc_opt = InConv(4, 64, DoubleConv)

        self.mmtm1 = self.mmtm1 = MMTM(64, 64, 1, device=device)

        self.max1_sar = nn.MaxPool2d(2)
        self.max1_opt = nn.MaxPool2d(2)

        self.conv1_sar = DoubleConv(64, 128)
        self.conv1_opt = DoubleConv(64, 128)

        self.mmtm2 = MMTM(128, 128, 1, device=device)

        self.max2_sar = nn.MaxPool2d(2)
        self.max2_opt = nn.MaxPool2d(2)

        self.conv2_sar = DoubleConv(128, 128)
        self.conv2_opt = DoubleConv(128, 128)

        self.mmtm3 = MMTM(128, 128, 1, device=device)

        self.up1_sar = nn.ConvTranspose2d(128, 128, (2, 2), stride=(2, 2))
        self.up1_opt = nn.ConvTranspose2d(128, 128, (2, 2), stride=(2, 2))

        self.conv3_sar = DoubleConv(256, 64)
        self.conv3_opt = DoubleConv(256, 64)

        self.mmtm4 = MMTM(64, 64, 1, device=device)

        self.up2_sar = nn.ConvTranspose2d(64, 64, (2, 2), stride=(2, 2))
        self.up2_opt = nn.ConvTranspose2d(64, 64, (2, 2), stride=(2, 2))

        self.conv4_sar = DoubleConv(128, 64)
        self.conv4_opt = DoubleConv(128, 64)

        self.outc_sar = OutConv(64, 1)
        self.outc_opt = OutConv(64, 1)

    def forward(self, x_sar, x_opt, curation_mode=False, caring_modality=None, rec_mmtm_squeeze=False):

        mmtm_kwargs = {
            'turnoff_cross_modal_flow': True if self.mmtm_off else False,
            'curation_mode': curation_mode,
            'caring_modality': caring_modality,
            'rec_mmtm_squeeze': rec_mmtm_squeeze,
        }

        features_sar = self.inc_sar(x_sar)
        features_opt = self.inc_opt(x_opt)

        features_sar, features_opt = self.mmtm1(features_sar, features_opt, **mmtm_kwargs)
        skip1_sar, skip1_opt = features_sar, features_opt

        features_sar = self.max1_sar(features_sar)
        features_sar = self.conv1_sar(features_sar)
        features_opt = self.max1_opt(features_opt)
        features_opt = self.conv1_opt(features_opt)

        features_sar, features_opt = self.mmtm2(features_sar, features_opt, **mmtm_kwargs)
        skip2_sar, skip2_opt = features_sar, features_opt

        features_sar = self.max2_sar(features_sar)
        features_sar = self.conv2_sar(features_sar)
        features_opt = self.max2_opt(features_opt)
        features_opt = self.conv2_opt(features_opt)

        features_sar, features_opt = self.mmtm3(features_sar, features_opt, **mmtm_kwargs)

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

        features_sar, features_opt = self.mmtm4(features_sar, features_opt, **mmtm_kwargs)

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

        return (out_sar + out_opt) / 2, [out_sar, out_opt]


@gin.configurable
class MMTM(nn.Module):
    def __init__(self, dim_sar, dim_opt, ratio, device='cuda'):
        super(MMTM, self).__init__()
        dim = dim_sar + dim_opt
        dim_out = int(2 * dim / ratio)

        self.ravg_out_sar = torch.zeros(dim_sar).to(device)
        self.ravg_out_opt = torch.zeros(dim_opt).to(device)
        self.step = 0

        self.ravg_squeeze_sar = nn.Parameter(torch.zeros((1, dim_sar), requires_grad=False))
        self.ravg_squeeze_opt = nn.Parameter(torch.zeros((1, dim_opt), requires_grad=False))
        self.rec_step = 0

        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_sar = nn.Linear(dim_out, dim_sar)
        self.fc_opt = nn.Linear(dim_out, dim_opt)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, sar, opt, turnoff_cross_modal_flow=False, curation_mode=False, caring_modality=0,
                rec_mmtm_squeeze=False):

        if not turnoff_cross_modal_flow:
            tview_sar = sar.view(sar.shape[:2] + (-1,))
            squeeze_sar = torch.mean(tview_sar, dim=-1)

            tview_opt = opt.view(opt.shape[:2] + (-1,))
            squeeze_opt = torch.mean(tview_opt, dim=-1)

            if rec_mmtm_squeeze:
                self.ravg_squeeze_sar = (squeeze_sar.mean(0) + self.ravg_sqeeze_sar * self.rec_step).detach()\
                                        / (self.rec_step + 1)
                self.ravg_squeeze_opt = (squeeze_opt.mean(0) + self.ravg_sqeeze_opt * self.rec_step).detach()\
                                        / (self.rec_step + 1)
                self.rec_step += 1

            squeeze = torch.cat((squeeze_sar, squeeze_opt), 1)
            excitation = self.fc_squeeze(squeeze)
            excitation = self.relu(excitation)

            sar_out = self.fc_sar(excitation)
            opt_out = self.fc_opt(excitation)

        else:
            tview = sar.view(sar.shape[:2] + (-1,))
            squeeze = torch.cat([torch.mean(tview, dim=-1),
                                 torch.stack(sar.shape[0] * [self.running_avg_squeeze_opt])], 1)
            excitation = self.relu(self.fc_squeeze(squeeze))

            sar_out = self.fc_sar(excitation)

            tview = opt.view(opt.shape[:2] + (-1,))
            squeeze = torch.cat([torch.stack(opt.shape[0] * [self.running_avg_squeeze_sar]),
                                 torch.mean(tview, dim=-1)], 1)
            excitation = self.relu(self.fc_squeeze(squeeze))

            opt_out = self.fc_opt(excitation)

        sar_out = self.sigmoid(sar_out)
        opt_out = self.sigmoid(opt_out)

        # running average weights of output
        self.ravg_out_sar = (sar_out.mean(0) + self.ravg_out_sar * self.step).detach() / (self.step + 1)
        self.ravg_out_opt = (opt_out.mean(0) + self.ravg_out_opt * self.step).detach() / (self.step + 1)
        self.step += 1

        if curation_mode:
            # for re-balancing steps, either one of the excitation signals is replaced with its respective avg weight
            if caring_modality == 0:
                sar_out = torch.stack(sar_out.shape[0] * [self.ravg_weight_sar])  # (B, C)

            elif caring_modality == 1:
                opt_out = torch.stack(opt_out.shape[0] * [self.ravg_weight_opt])

        # matching the shape of the excitation signals to the input features for recalibration
        # (B, C) -> (B, C, H, W)
        sar_out = sar_out.view(sar_out.shape + (1,) * (len(sar.shape) - len(sar_out.shape)))
        opt_out = opt_out.view(opt_out.shape + (1,) * (len(opt.shape) - len(opt_out.shape)))

        return sar * sar_out, opt * opt_out


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


