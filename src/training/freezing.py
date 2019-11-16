from imports import *


def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        m.weight.requires_grad = False
        m.bias.requires_grad = False


def unfreeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()
        m.weight.requires_grad = True
        m.bias.requires_grad = True


def freeze(m):
    for param in m.parameters():
        param.requires_grad = False
    unfreeze_bn(m)


def unfreeze(m):
    for param in m.parameters():
        param.requires_grad = True


def freeze_encoder(model):
    for n, p in list(model.named_parameters()):
        if "encoder" in n:
            p.requires_grad = False
    model.apply(unfreeze_bn)


def unfreeze_encoder(model):
    for n, p in list(model.named_parameters()):
        if "encoder" in n:
            p.requires_grad = True


def requires_grad(m: nn.Module, b: Optional[bool] = None)->Optional[bool]:
    """If `b` is not set return `requires_grad` of first param, else set `requires_grad` on all params as `b`"""
    ps = list(m.parameters())
    if not ps:
        return None
    if b is None:
        return ps[0].requires_grad
    for p in ps:
        p.requires_grad = b


def freeze_to(layer_groups, n: int)->None:
    """
    Freeze layers up to layer group `n`.
    """
    for g in layer_groups[:n]:
        for l in g:
            if not (
                isinstance(
                    l,
                    nn.BatchNorm2d) or isinstance(
                    l,
                    nn.BatchNorm1d)):
                requires_grad(l, False)
    for g in layer_groups[n:]:
        requires_grad(g, True)

