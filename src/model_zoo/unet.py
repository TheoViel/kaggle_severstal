from imports import *
from model_zoo.common import *
from custom_layers.scse import *
from custom_layers.aspp import *


class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, **batchnorm_params):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm),
            nn.ReLU(inplace=True), ]
        if use_batchnorm:
            layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        if attention_type is None:
            self.attention1 = nn.Identity()
            self.attention2 = nn.Identity()
        elif attention_type == 'scse':
            self.attention1 = SCSEModule(in_channels)
            self.attention2 = SCSEModule(out_channels)

        self.block = nn.Sequential(
            Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
            Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
        )

    def forward(self, x):
        x, skip = x

        x = interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)

        x = self.block(x)
        x = self.attention2(x)
        return x


class CenterBlock(DecoderBlock):
    def forward(self, x):
        return self.block(x)


class UnetDecoder(Model):
    def __init__(self, encoder_channels, decoder_channels=(64, 64, 64, 64), final_channels=1,
                 use_batchnorm=True, attention_type=None):
        super().__init__()

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm,
                                   attention_type=attention_type)
        self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm,
                                   attention_type=attention_type)
        self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm,
                                   attention_type=attention_type)
        self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm,
                                   attention_type=attention_type)

        global_conv_size = 64
        self.global_layer = nn.Sequential(
            nn.Conv2d(np.sum(out_channels), global_conv_size, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(global_conv_size),

            nn.Conv2d(global_conv_size, global_conv_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(global_conv_size),
        )

        self.final_conv = nn.Conv2d(global_conv_size, final_channels, kernel_size=1)

        self.initialize()

    @staticmethod
    def compute_channels(encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        x1 = self.layer1([encoder_head, skips[0]])
        x2 = self.layer2([x1, skips[1]])
        x3 = self.layer3([x2, skips[2]])
        x4 = self.layer4([x3, skips[3]])

        h, w = x4.size()[2:]
        x = torch.cat([
            F.upsample_bilinear(x1, size=(h, w)),
            F.upsample_bilinear(x2, size=(h, w)),
            F.upsample_bilinear(x3, size=(h, w)),
            x4
        ], 1)

        x = self.global_layer(x)
        x = F.upsample_bilinear(x, size=(2 * h, 2 * w))

        return self.final_conv(x)


class SegmentationUnet(Model):
    def __init__(self, encoder_settings, num_classes=4, center_block=None, aux_clf=False,
                 use_bn=True, attention_type=None):
        super().__init__()
        self.aux_clf = aux_clf
        self.num_classes = num_classes

        self.encoder = get_encoder(encoder_settings)
        encoder_chanels = list(self.encoder.out_shapes).copy()

        if center_block == "aspp":
            self.center = ASPP(self.encoder.out_shapes[0], self.encoder.out_shapes[1])
            encoder_chanels[0] = self.encoder.out_shapes[1]
        elif center_block == "std":
            self.center = CenterBlock(self.encoder.out_shapes[0], self.encoder.out_shapes[0], use_batchnorm=use_bn)
        else:
            self.center = None

        self.decoder = UnetDecoder(encoder_channels=encoder_chanels, final_channels=num_classes + 1,
                                   use_batchnorm=use_bn, attention_type=attention_type)

        self.logit = nn.Sequential(nn.Conv2d(encoder_chanels[0] * 2, 32, kernel_size=1),
                                   nn.Conv2d(32, num_classes, kernel_size=1))

    def forward(self, x):
        head, x3, x2, x1, x0 = self.encoder(x)

        if self.center is not None:
            head = self.center(head)

        masks = self.decoder([head, x3, x2, x1, x0])

        if self.aux_clf:
            x = adaptive_concat_pool2d(head)
            return masks, self.logit(x).view(-1, self.num_classes)

        return masks, _


class UnetBlock(Module):
    "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

    def __init__(self, up_in_c: int, x_in_c: int, final_div: bool = True, blur: bool = False, leaky: float = None,
                 self_attention: bool = False, **kwargs):
        super(UnetBlock, self).__init__()
        self.reduce_conv = conv_layer(x_in_c, x_in_c // 4, leaky=leaky, norm_type=None, **kwargs)
        x_in_c = x_in_c // 4
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, leaky=leaky, **kwargs)
        self.bn = batchnorm_2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = ni if final_div else ni // 2
        self.conv1 = conv_layer(ni, nf, leaky=leaky, norm_type=None, **kwargs)
        self.conv2 = conv_layer(nf, nf, leaky=leaky, norm_type=None, self_attention=self_attention, **kwargs)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor, left_in: Tensor) -> Tensor:
        s = self.reduce_conv(left_in)
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))


class UnetBlockI(Module):
    def __init__(self, up_in_c: int, x_in_c: int, nf: int = None, blur: bool = False, leaky: float = None,
                 self_attention: bool = False, **kwargs):
        super(UnetBlockI, self).__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c // 2, blur=blur, leaky=leaky, **kwargs)
        # self.bn = batchnorm_2d(x_in_c)#swithch in 2
        self.bn = GBnorm_2d(x_in_c)
        ni = up_in_c // 2 + x_in_c
        nf = nf if nf is not None else max(up_in_c // 2, 32)
        self.conv1 = conv_layer(ni, nf, leaky=leaky, norm_type=None, **kwargs)
        self.conv2 = conv_layer(nf, nf, leaky=leaky, norm_type=None, self_attention=self_attention, **kwargs)
        self.relu = relu(leaky=leaky)

    def forward(self, up_in: Tensor, left_in: Tensor) -> Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))

