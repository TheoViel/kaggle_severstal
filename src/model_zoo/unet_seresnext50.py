from model_zoo.unet import *
from custom_layers.mish import *
from custom_layers.hypercolumns import *

import pretrainedmodels


class UneXt50SE(nn.Module):
    def __init__(self, encoder_settings, num_classes=4, pre=False, **kwargs):
        super(UneXt50SE, self).__init__()
        self.n_classes = num_classes
        m = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet' if pre else None)

        # conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # w = (m.layer0.conv1.weight.sum(1)).unsqueeze(1)
        # conv.weight = nn.Parameter(w)
        # self.enc0 = nn.Sequential(conv, m.layer0.bn1, nn.ReLU(inplace=True))

        self.enc0 = nn.Sequential(m.layer0.conv1, m.layer0.bn1, nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.enc1 = m.layer1  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        self.middle_conv = ASPPI(2048, 1024)
        self.dec4 = UnetBlockI(1024, 1024, 512)
        self.dec3 = UnetBlockI(512, 512, 256)
        self.dec2 = UnetBlockI(256, 256, 128)
        self.dec1 = UnetBlockI(128, 64, 64)
        self.hc = HyperColumnI([1024, 512, 256, 128], [32] * 4)
        self.drop = nn.Dropout2d(0.2)
        self.final_conv = conv_layer(64 + 32 * 4, self.n_classes + 1, ks=1, norm_type=None, use_activ=False)
        self.logit = nn.Sequential(nn.Dropout2d(0.5), nn.Conv2d(5120 * 2, 64, kernel_size=1), nn.ReLU(inplace=True),
                                   GBnorm_2d(64), nn.Conv2d(64, self.n_classes, kernel_size=1))

        to_Mish(self.dec4), to_Mish(self.dec3), to_Mish(self.dec2), to_Mish(self.dec1)
        to_Mish(self.final_conv), to_Mish(self.middle_conv)
        apply_init(self.middle_conv, nn.init.kaiming_normal_)
        apply_init(self.dec4, nn.init.kaiming_normal_)
        apply_init(self.dec3, nn.init.kaiming_normal_)
        apply_init(self.dec2, nn.init.kaiming_normal_)
        apply_init(self.dec1, nn.init.kaiming_normal_)
        apply_init(self.final_conv, nn.init.kaiming_normal_)
        apply_init(self.hc, nn.init.kaiming_normal_)

    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(self.pool(enc0))
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5, x_aux = self.middle_conv(enc4)
        dec3 = self.dec4(enc5, enc3)
        dec2 = self.dec3(dec3, enc2)
        dec1 = self.dec2(dec2, enc1)
        dec0 = self.dec1(dec1, enc0)
        x = self.hc([enc5, dec3, dec2, dec1], dec0)
        x = self.final_conv(self.drop(x))
        x = F.upsample_bilinear(x, scale_factor=(2, 2))
        x_aux = self.logit(x_aux).view(-1, self.n_classes)
        return x, x_aux