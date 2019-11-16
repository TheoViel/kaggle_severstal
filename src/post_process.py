from training.predicting import *
from metric import *


def plot_confusion_matrix(y_true, y_pred, classes=['0', '1'], normalize=False, title="", cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]), 
           yticks=[], #np.arange(cm.shape[1]), 
           xticklabels=classes, 
           yticklabels=classes,
           title=title, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()


def predict_faults(dataset, model, batch_size=VAL_BS, tta=False):
    model.eval()
    preds = np.array([[], [], [], []]).T
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    with torch.no_grad():
        for x, truth, fault, _ in tqdm(loader):

            _, prob = model(x.to(DEVICE))
            probs = torch.sigmoid(prob.detach()).cpu().numpy()

            if tta:
                flips = [[-1], [-2], [-2, -1]]
                for f in flips:
                    _, prob = model(torch.flip(x.to(DEVICE), f))
                    probs += torch.sigmoid(prob.detach()).cpu().numpy()
                probs /= len(flips) + 1
            preds = np.concatenate([preds, probs])
    return preds


def tweak_thresholds_clf(pred, truth):
    thresholds = []
    for i in range(4):
        best_score = 0
        best_t = 0
        for t in np.arange(0.2, 0.7, 0.01):
            score = accuracy_score((truth[: , i] > 0).astype(int), (pred[:, i] > t).astype(int))
            if score > best_score:
                best_t = t
                best_score = score
        thresholds.append(best_t)
    return np.round(np.array(thresholds), 2)


def post_process(mask_prob, threshold=0.5, min_size=100):
    mask = (mask_prob > threshold).astype(np.uint8)
    num_component, components = cv2.connectedComponents(mask, connectivity=8)
    processed_mask = np.zeros(mask_prob.shape, np.float32)

    num = 0
    for c in range(1, num_component):
        p = (components == c)
        if p.sum() > min_size:
            processed_mask[p] = 1
            num += 1
    return processed_mask


def get_best_params(dices, thresholds, min_sizes):
    best_params = []
    best_dices = []

    for c in range(4):
        best_threshold = 0.5
        best_min_size = 0
        best_dice = 0
        for threshold in thresholds:
            for min_size in min_sizes:
                dice = dices[c][threshold][min_size]
                if dice > best_dice:
                    best_dice = dice
                    best_threshold = threshold
                    best_min_size = min_size
        best_params.append([best_threshold, best_min_size])
        best_dices.append(best_dice)

    print(f'Local validation dice is {np.mean(best_dices):.4f}\n')  # 0.3; 0.3
    for i, d in enumerate(best_dices):
        print(f' -> Class {i + 1} : {best_dices[i]:.4f}')

    best_thresholds = np.array(best_params)[:, 0]
    best_min_sizes = np.array(best_params)[:, 1]

    return best_thresholds, best_min_sizes


def tweak_thresholds(dataset, models, clf_probs, batch_size=4, thresholds=(0.5, 0.5, 0.5, 0.5), min_sizes=(0, 0, 0, 0),
                     prob_thresholds=(0, 0, 0, 0), prob_thresholds_aux=(0, 0, 0, 0), prob_thresholds_max=(0, 0, 0, 0),
                     tta=False):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    dices = {i:
                 {threshold: {min_size: 0 for min_size in min_sizes} for threshold in thresholds}
             for i in range(4)}

    for idx, (x, _, truth, fault) in enumerate(tqdm(loader)):
        masks_ = []
        probs_ = []
        for model in models:
            masks, probs = predict_seg_softmax(model, x, tta=tta)
            masks_.append(masks)
            probs_.append(probs)

        masks = np.mean(np.array(masks_), axis=0)[-4:]

        probs_max = np.max(np.max(masks, axis=-1), axis=-1)
        probs_aux = np.mean(np.array(probs_), axis=0).T
        probs = clf_probs[batch_size * idx: min(batch_size * (idx + 1), clf_probs.shape[0]), :].T

        for threshold in thresholds:
            for min_size in min_sizes:
                for i, mask in enumerate(masks):
                    processed_masks = []
                    for j, m in enumerate(mask):
                        if probs[i, j] < prob_thresholds[i]:
                            processed_masks.append(np.zeros((IMG_SHAPE[0], IMG_SHAPE[1])))
                        elif probs_aux[i, j] < prob_thresholds_aux[i]:
                            processed_masks.append(np.zeros((IMG_SHAPE[0], IMG_SHAPE[1])))
                        elif probs_max[i, j] < prob_thresholds_max[i]:
                            processed_masks.append(np.zeros((IMG_SHAPE[0], IMG_SHAPE[1])))
                        else:
                            if min_size > 0:
                                processed_masks.append(post_process(m, threshold=threshold, min_size=min_size))
                            else:
                                processed_masks.append((m > threshold).astype(int))

                    processed_masks = np.array(processed_masks)

                    dices[i][threshold][min_size] += dice_np(
                        processed_masks.reshape((-1, 1, IMG_SHAPE[0], IMG_SHAPE[1])),
                        truth.numpy()[:, i, :, :].reshape(
                            (-1, 1, IMG_SHAPE[0], IMG_SHAPE[1]))
                        ) / len(loader)
    return dices
