from params import *
from data.masks import *


def dice_np(pred, truth, eps=1e-8, threshold=0.5):
    n, c = truth.shape[0], pred.shape[1]
    pred = (pred.reshape((n * c, -1)) > threshold).astype(int)
    truth = truth.reshape((n * c, -1))

    intersect = (pred + truth == 2).sum(-1)
    union = pred.sum(-1) + truth.sum(-1)

    return ((2.0 * intersect + eps) / (union + eps)).mean()


def dice_th(pred, truth, eps=1e-8, threshold=0.5):
    n, c = truth.shape[0], pred.shape[1]
    pred = (pred.view(n * c, -1) > threshold).float()
    truth = truth.view(n * c, -1)

    intersect = (pred + truth == 2).sum(-1).float()
    union = pred.sum(-1) + truth.sum(-1).float()

    return ((2.0 * intersect + eps) / (union + eps)).mean()


def eval_predictions(dataset, rles_pred):
    dice = 0
    for i in tqdm(range(len(dataset))):
        img, _, truth, fault = dataset[i]
        pred = np.array([rle_to_mask(rle, IMG_SHAPE) for rle in rles_pred[4*i: 4*(i+1)]])
        assert truth.shape == pred.shape
        dice += dice_np(np.array([pred]), np.array([truth])) / len(dataset)
    return dice
