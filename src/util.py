from imports import *
from fastai.vision import *
from params import *


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # False


def save_model_weights(model, filename, verbose=1, cp_folder=CP_PATH):
    if verbose:
        print(f'\n -> Saving weights to {os.path.join(cp_folder,filename)}\n')
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def load_model_weights(
        model,
        filename,
        verbose=1,
        cp_folder=CP_PATH,
        strict=True):
    if verbose:
        print(
            f'\n -> Loading weights from {os.path.join(cp_folder,filename)}\n')
    try:
        model.load_state_dict(os.path.join(cp_folder, filename), strict=strict)
    except BaseException:
        model.load_state_dict(
            torch.load(
                os.path.join(
                    cp_folder,
                    filename),
                map_location='cpu'),
            strict=strict)
    return model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def count_parameters(model, all=False):
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_model(m): return sum(
    map(flatten_model, children_and_parameters(m)), []) if num_children(m) else [m]


def listify(p: OptListOrItem = None, q: OptListOrItem = None):
    """
    Make `p` listy and the same length as `q`.
    """
    if p is None:
        p = []
    elif isinstance(p, str):
        p = [p]
    elif not isinstance(p, Iterable):
        p = [p]
    # Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try:
            _ = len(p)
        except BaseException:
            p = [p]
    n = q if isinstance(q, int) else len(p) if q is None else len(q)
    if len(p) == 1:
        p = p * n
    assert len(p) == n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


def first_layer(m: nn.Module)->nn.Module:
    """"
    Retrieve first layer in a module `m`.
    """
    return flatten_model(m)[0]


def split_model_idx(model: nn.Module, idxs: Collection[int])->ModuleList:
    """
    Split `model` according to the indexes in `idxs`.
    """
    layers = flatten_model(model)
    if idxs[0] != 0:
        idxs = [0] + idxs
    if idxs[-1] != len(layers):
        idxs.append(len(layers))
    return [nn.Sequential(*layers[i:j]) for i, j in zip(idxs[:-1], idxs[1:])]


def split_model(model: nn.Module = None,
                splits: Collection[Union[nn.Module, ModuleList]] = None):
    """
    Split `model` according to the layers in `splits`.
    """
    splits = listify(splits)
    if isinstance(splits[0], nn.Module):
        layers = flatten_model(model)
        idxs = [layers.index(first_layer(s)) for s in splits]
        return split_model_idx(model, idxs)
    return [nn.Sequential(*s) for s in splits]
