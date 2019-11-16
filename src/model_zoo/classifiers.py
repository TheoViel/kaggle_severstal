import pretrainedmodels
from model_zoo.fpn_enetb5 import *


class Classifier(Model):
    def __init__(self, encoder_settings, num_classes=4, center_block=None, drop_connect_rate=0.2,
                 use_bn=True):
        super().__init__()
        self.num_classes = num_classes

        try:
            self.encoder = get_encoder(encoder_settings)
        except:
            self.encoder = EfficientNetB5(drop_connect_rate)
            load_pretrain(self.encoder, pretrain_file='efficientnet-b5-b6417697.pth')

        encoder_chanels = list(self.encoder.out_shapes).copy()

        if center_block == "aspp":
            self.center = ASPP(self.encoder.out_shapes[0], self.encoder.out_shapes[1])
            encoder_chanels[0] = self.encoder.out_shapes[1]
        elif center_block == "std":
            self.center = CenterBlock(self.encoder.out_shapes[0], self.encoder.out_shapes[0], use_batchnorm=use_bn)
        else:
            self.center = None

        self.dropout = nn.Dropout(p=0.5)

        self.logit = nn.Sequential(nn.Conv2d(encoder_chanels[0] * 2, 32, kernel_size=1),
                                   nn.Conv2d(32, num_classes, kernel_size=1))

    def forward(self, x):
        head, x3, x2, x1, x0 = self.encoder(x)

        if self.center is not None:
            head = self.center(head)

        x = adaptive_concat_pool2d(self.dropout(head))
        #         x = adaptive_concat_pool2d(head)
        return self.logit(x).view(-1, self.num_classes)


class UneXt50SE_c(nn.Module):
    def __init__(self, encoder_settings, num_classes=4, seql=64, pre=True, **kwargs):
        super(UneXt50SE_c, self).__init__()
        self.n_classes = num_classes
        m = pretrainedmodels.se_resnext50_32x4d(pretrained='imagenet' if pre else None)

        self.enc0 = nn.Sequential(m.layer0.conv1, m.layer0.bn1, nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=True)
        self.enc1 = m.layer1  # 256
        self.enc2 = m.layer2  # 512
        self.enc3 = m.layer3  # 1024
        self.enc4 = m.layer4  # 2048
        self.middle_conv = ASPPI(2048, 1024)
        self.logit = nn.Sequential(nn.Dropout2d(0.5), nn.Conv2d(5120 * 2, 64, kernel_size=1), nn.ReLU(inplace=True),
                                   GBnorm_2d(64), nn.Conv2d(64, self.n_classes, kernel_size=1))
        self.drop = nn.Dropout2d(0.2)
        self.final_conv_c = conv_layer(1024, self.n_classes + 1, ks=3, norm_type=None, use_activ=False)

        to_Mish(self.final_conv_c), to_Mish(self.middle_conv)
        apply_init(self.final_conv_c, nn.init.kaiming_normal_)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = F.upsample_bilinear(x, size=(h // 2, w // 2))
        enc0 = self.enc0(x)
        enc1 = self.enc1(self.pool(enc0))
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5, x_aux = self.middle_conv(enc4)
        x = self.final_conv_c(self.drop(enc5))
        x_aux = self.logit(x_aux).view(-1, self.n_classes)
        return x, x_aux   
