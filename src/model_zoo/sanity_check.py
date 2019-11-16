from params import *
from model_zoo.unet import *
from model_zoo.fpn_enetb5 import *
from model_zoo.common import SETTINGS
from model_zoo.unet_seresnext50 import *
from model_zoo.unet_densenet169 import *


if __name__ == '__main__':
    print('Building Unet with Resnet34 backbone...')
    _ = SegmentationUnet(SETTINGS['resnet34'], num_classes=4, center_block="aspp", aux_clf=True).to(DEVICE)
    print('Building Unet with SeResNext50 backbone...')
    _ = SegmentationUnet(SETTINGS['se_resnext50_32x4d'], num_classes=4, center_block="aspp", aux_clf=True).to(DEVICE)
    print('Building custom Unet with SeResNext50 backbone...')
    _ = UneXt50SE(_, num_classes=4, pre=False).to(DEVICE)
    print('Building custom Unet with DenseNet169 backbone...')
    _ = UDnet169(_, num_classes=4, pre=False).to(DEVICE)
    print('Building custom FPN with EfficientNetB5 backbone...')
    _ = SegmentationFPNEfficientNet(num_classes=4, center_block="aspp", aux_clf=True).to(DEVICE)