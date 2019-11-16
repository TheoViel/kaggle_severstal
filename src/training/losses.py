from imports import *

__all__ = [
    'hck_focal_loss',
    'symmetric_lovasz',
    'lov_loss',
    'criterion_mix',
    'bce_r',
    'acc_r',
    'acc']


def hck_focal_loss(logit, truth, weight=(1, 1, 1, 1), alpha=2):
    weight = torch.FloatTensor([1] + weight).to(truth.device).view(1, 1, -1)

    batch_size, num_class, H, W = logit.shape

    logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, 5)
    truth = truth.contiguous().view(-1)

    log_probability = -F.log_softmax(logit, -1)
    probability = F.softmax(logit, -1)

    onehot = torch.zeros(batch_size * H * W, num_class).to(truth.device)
    onehot.scatter_(dim=1, index=truth.view(-1, 1), value=1)

    loss = log_probability * onehot

    # ---
    if 1:  # image based focusing
        probability = probability.view(batch_size, H * W, 5)
        truth = truth.view(batch_size, H * W, 1)

        focal = torch.gather(
            probability,
            dim=-1,
            index=truth.view(
                batch_size,
                H * W,
                1))
        focal = (1 - focal) ** alpha
        focal_sum = focal.sum(dim=[1, 2], keepdim=True)

        weight = weight * focal / focal_sum.detach() * H * W
        weight = weight.view(-1, 5)

    loss = loss * weight
    loss = loss.mean()
    return loss


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    # loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
    return loss


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(
            lovasz_hinge_flat(
                *
                flatten_binary_scores(
                    log.unsqueeze(0),
                    lab.unsqueeze(0),
                    ignore)) for log,
            lab in zip(
                logits,
                labels))
    else:
        loss = lovasz_hinge_flat(
            *
            flatten_binary_scores(
                logits,
                labels,
                ignore))
    return loss


def symmetric_lovasz(outputs, targets):
    targets = targets.float()
    return (lovasz_hinge(outputs, targets) +
            lovasz_hinge(-outputs, 1 - targets)) / 2


def lov_loss(x, y):
    L = symmetric_lovasz
    return (L(x[:, 0].unsqueeze(1), y == 0) +
            L(x[:, 1].unsqueeze(1), y == 1) +
            L(x[:, 2].unsqueeze(1), y == 2) +
            L(x[:, 3].unsqueeze(1), y == 3) +
            L(x[:, 4].unsqueeze(1), y == 4)) / 5.0


def criterion_mix(logit, truth):
    return lov_loss(logit, truth) + hck_focal_loss(logit, truth)


def reduce_mask(mask, r=64):
    bs, c, w, h = mask.shape
    m = mask.view(-1, 5, w // r, r, h // r, r).max(5)[0].max(3)[0]
    m[:, 0, ...] = 1 - m[:, 1:, ...].max(1)[0]
    return m


def bce_r(x, y):
    # print(y.shape,y.max())
    return nn.BCEWithLogitsLoss()(x, reduce_mask(y).float())


def acc_r(x, y):
    x = x[:, 1:, ...].max(-1)[0].max(-1)[0]
    y = y[:, 1:, ...].max(-1)[0].max(-1)[0].byte()
    return ((x > 0.0) == y).float().mean()


def acc(x, y):
    return ((x > 0.0) == y.byte()).float().mean()
