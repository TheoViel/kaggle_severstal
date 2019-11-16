from imports import *


class HyperColumn(Module):
    def __init__(self, input_channels: list, output_channels: list, im_size: int, kernel_size=1):
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
             for in_ch, out_ch in zip(input_channels, output_channels)])

    def forward(self, xs: list, last_layer=None):
        bs, ch, *image_size = last_layer.shape
        up = nn.Upsample(image_size, mode='bilinear')
        hcs = [up(c(x)) for c, x in zip(self.convs, xs)]
        if last_layer is not None:
            hcs.append(last_layer)
        return torch.cat(hcs, dim=1)


class HyperColumnI(nn.Module):
    def __init__(self, input_channels: list, output_channels: list):
        super(HyperColumnI, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch * 2, kernel_size=3, padding=1),
                           nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, padding=1))
             for in_ch, out_ch in zip(input_channels, output_channels)])
        # self.up = nn.Upsample(im_size, mode='bilinear', align_corners=False)

    def forward(self, xs: list, last_layer):
        hcs = [F.interpolate(c(x), scale_factor=2 ** (len(self.convs) - i))
               for i, (c, x) in enumerate(zip(self.convs, xs))]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)