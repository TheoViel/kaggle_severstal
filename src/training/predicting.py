from util import *
from params import *
from imports import *
from data.masks import *
from post_process import *
from data.dataset import *

def predict_seg_softmax(model, x, tta=False, t=1):
    model.eval()
    with torch.no_grad():
        masks, prob = model(x.to(DEVICE))
        bs, c, h, w = masks.size()

        masks = nn.Softmax(-1)(masks.permute(0, 2, 3, 1).reshape(-1, c).detach()).reshape(bs, h, w, c)
        masks = masks.permute(0, 3, 1, 2).contiguous() ** t
        probs = torch.sigmoid(prob.detach()).cpu().numpy() ** t

        if tta:
            flips = [[-1], [-2], [-2, -1]]
#             flips = [[-1], [-2]]
            for f in flips:
                mask, prob = model(torch.flip(x.to(DEVICE), f))

                m = nn.Softmax(-1)(torch.flip(mask, f).permute(0, 2, 3, 1).reshape(-1, c).detach()).reshape(bs, h, w, c)
                masks += m.permute(0, 3, 1, 2).contiguous() ** t

                probs += torch.sigmoid(prob.detach()).cpu().numpy() ** t

            masks /= len(flips) + 1
            probs /= len(flips) + 1

        masks = masks.cpu().numpy().transpose(1, 0, 2, 3) # class first
    return masks, probs.tolist()


def predict(dataset, models, prob_clf, batch_size=VAL_BS, thresholds=(0.5, 0.5, 0.5, 0.5), min_sizes=(0, 0, 0, 0),
            prob_thresholds=(0, 0, 0, 0), prob_thresholds_aux=(0, 0, 0, 0), prob_thresholds_max=(0, 0, 0, 0),
            tta=False, t=1):
    rles = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    all_probs_aux = np.array([[], [], [], []]).T
    all_probs_max = np.array([[], [], [], []]).T

    with torch.no_grad():
        for idx, (x, _, truth, fault) in enumerate(tqdm(loader)):
            rles_batch = []
            masks_ = []
            probs_ = []

            for model in models:
                masks, probs = predict_seg_softmax(model, x, tta=tta)
                masks_.append(masks)
                probs_.append(probs)

            masks = np.mean(np.array(masks_) ** t, axis=0)[-4:]

            probs_max = np.max(np.max(masks, axis=-1), axis=-1)
            probs_aux = np.mean(np.array(probs_), axis=0).T
            probs = prob_clf[batch_size * idx: min(batch_size * (idx + 1), prob_clf.shape[0]), :].T

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
                        if min_sizes[i] > 0:
                            processed_masks.append(post_process(m, threshold=thresholds[i], min_size=min_sizes[i]))
                        else:
                            processed_masks.append((m > thresholds[i]).astype(int))

                processed_masks = np.array(processed_masks)
                rles_batch += [mask_to_rle(mask) for mask in processed_masks]

            rles_batch = np.array(rles_batch).reshape((4, -1))
            for i in range(rles_batch.shape[-1]):
                rles += list(rles_batch[:, i])
            all_probs_aux = np.concatenate((all_probs_aux, probs_aux.T))
            all_probs_max = np.concatenate((all_probs_max, probs_max.T))

    return rles, all_probs_aux, all_probs_max


def predict_vote(dataset, models, prob_clf, thresholds, min_sizes,
                 prob_thresholds, prob_thresholds_aux, prob_thresholds_max,
                 tta=False, t=1, batch_size=VAL_BS):
    rles = []
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    for idx, (x, _, truth, fault) in enumerate(tqdm(loader)):
        rles_batch = []
        all_masks = []

        probs = prob_clf[batch_size * idx: min(batch_size * (idx + 1), prob_clf.shape[0]), :].T

        for idx, model in enumerate(models):
            masks_ = []
            probs_ = []

            masks, probs_aux = predict_seg_softmax(model, x, tta=tta)
            masks = masks[-4:]
            probs_aux = np.array(probs_aux).T
            probs_max = np.max(np.max(masks, axis=-1), axis=-1)

            for i, mask in enumerate(masks):
                processed_masks = []
                for j, m in enumerate(mask):
                    if probs[i, j] < prob_thresholds[i]:
                        processed_masks.append(np.zeros((IMG_SHAPE[0], IMG_SHAPE[1])))
                    elif probs_aux[i, j] < prob_thresholds_aux[idx, i]:
                        processed_masks.append(np.zeros((IMG_SHAPE[0], IMG_SHAPE[1])))
                    elif probs_max[i, j] < prob_thresholds_max[i]:
                        processed_masks.append(np.zeros((IMG_SHAPE[0], IMG_SHAPE[1])))
                    else:
                        if min_sizes[idx, i] > 0:
                            processed_masks.append(post_process(m, threshold=thresholds[idx, i],
                                                                min_size=min_sizes[idx, i]))
                        else:
                            processed_masks.append((m > thresholds[idx, i]).astype(int))
                masks_.append(np.array(processed_masks))
                probs_.append(probs)

            all_masks.append(np.array(masks_))

        masks = (np.mean(np.array(all_masks) ** t, axis=0) > 0.5).astype(int)

        #         reprocessed_masks = []
        #         for i, mask in enumerate(masks):
        #             processed_masks = []
        #             for j, m in enumerate(mask):
        #                 processed_masks.append(post_process(m, threshold=0.5, min_size=100))
        #             reprocessed_masks.append(np.array(processed_masks))

        #         masks = np.array(reprocessed_masks)

        rles_batch += [mask_to_rle(mask) for mask in masks.reshape((-1, 256, 1600))]
        rles_batch = np.array(rles_batch).reshape((4, -1))

        for i in range(rles_batch.shape[-1]):
            rles += list(rles_batch[:, i])

    return rles


def predict_faults(dataset, model, batch_size=VAL_BS, tta=False):
    model.eval()
    preds = np.array([[], [], [], []]).T
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    with torch.no_grad():
        for x, _ in tqdm(loader):
            prob = model(x.to(DEVICE))
            probs = torch.sigmoid(prob.detach()).cpu().numpy()

            if tta:
                flips = [[-1], [-2], [-2, -1]]
                for f in flips:
                    prob = model(torch.flip(x.to(DEVICE), f))
                    probs += torch.sigmoid(prob.detach()).cpu().numpy()
                probs /= len(flips) + 1
            preds = np.concatenate([preds, probs])
    return preds


def predict_faults_with_aux(dataset, model, batch_size=VAL_BS, tta=False):
    model.eval()
    preds = np.array([[], [], [], []]).T
    preds_max = np.array([[], [], [], []]).T
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    with torch.no_grad():
        for x, _ in tqdm(loader):
            y_aux, prob = model(x.to(DEVICE))
    
            probs_max = sigmoid(y_aux.detach().cpu().numpy().max(-1).max(-1)[:, 1: ,...])
            probs = torch.sigmoid(prob.detach()).cpu().numpy()

            if tta:
                flips = [[-1], [-2], [-2, -1]]
                for f in flips:
                    y_aux, prob = model(torch.flip(x.to(DEVICE), f))
                    probs_max += sigmoid(y_aux.detach().cpu().numpy().max(-1).max(-1)[:, 1: ,...])
                    probs += torch.sigmoid(prob.detach()).cpu().numpy()
                probs /= len(flips) + 1
                probs_max /= len(flips) + 1
                
            preds = np.concatenate([preds, probs])
            preds_max = np.concatenate([preds_max, probs_max])
            
    return preds, preds_max


def k_fold_predictions_clf(models, mask_dic, all_images, classes, classes_max, transforms, k=5, tta=False):
    splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(all_images, classes_max))
    preds = np.zeros((len(classes), 4))
    for i in range(len(models)):
        val_idx = splits[i][1]
        dataset = SteelValDatasetClf(mask_dic, all_images[val_idx], classes[val_idx], transforms)
        preds[val_idx, :] = predict_faults(dataset, models[i], tta=tta)
    return preds