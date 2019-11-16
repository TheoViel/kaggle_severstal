from pretrainedmodels.models.senet import SENet, SEBottleneck, SEResNetBottleneck, SEResNeXtBottleneck


class SENetEncoder(SENet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained = False

        del self.last_linear
        del self.avg_pool

    def forward(self, x):
        for module in self.layer0[:-1]:
            x = module(x)

        x0 = x
        x = self.layer0[-1](x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = [x4, x3, x2, x1, x0]
        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        super().load_state_dict(state_dict, **kwargs)