from metric import *
from params import *
from imports import *


def rle_to_mask(rle, shape):
    if rle == '-1':
        return np.zeros(shape)

    width = shape[0]
    height = shape[1]

    mask = np.zeros(width * height).astype(np.uint8)

    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        mask[int(start):int(start + lengths[index])] = 1
        current_position += lengths[index]

    return np.flipud(np.rot90(mask.reshape(height, width), k=1))


def mask_to_rle(img):
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_masks(img_name, mask_dic):
    return np.array([rle_to_mask(mask_dic[img_name][c], IMG_SHAPE)
                     for c in ['0', '1', '2', '3']]).transpose(1, 2, 0)


def plot_masks(img, masks):
    if masks.shape[0] == 5:
        masks = masks[1:, :, :]
    plt.figure(figsize=(15, 5))
    img = (img - img.min()) / (img.max() - img.min())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(4):
        if i < 3:
            img[masks[i] == 1, i] = 255
        else:
            img[masks[i] == 1, 0] = 200
            img[masks[i] == 1, 1] = 200

    faults = np.sum(np.sum(masks, axis=-1), axis=-1)
    plt.title(f'Pixel classes of sample : {faults}')

    plt.axis('off')
    plt.imshow(img)
    plt.show()

    
def viz_predictions(dataset, rles_pred, classes, n=1, plot_classes=[0, 1, 2, 3, 4]):
    count = 0
    while count < n:
        i = np.random.choice(len(dataset))
        if classes[i] in plot_classes:
            count += 1
            img, _, truth, _ = dataset[i]

            img = img[0]
            img = (img - img.min()) / (img.max() - img.min())
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs = [img, np.copy(img)]

            pred = np.array([rle_to_mask(rle, IMG_SHAPE) for rle in rles_pred[4*i: 4*(i+1)]])
            assert truth.shape == pred.shape

            plt.figure(figsize=(15, 5))
            for k, masks in enumerate([truth, pred]):
                plt.subplot(2, 1, k+1)
                for i in range(4):
                    if i < 3:
                        imgs[k][masks[i] == 1, i] = 255
                    else:
                        imgs[k][masks[i] == 1, 0] = 200
                        imgs[k][masks[i] == 1, 1] = 200

                faults = np.sum(np.sum(masks, axis=-1), axis=-1)
                prefix = 'Pred' if k else 'Truth'
                plt.title(f'{prefix} - Pixel classes of sample : {faults}')
                plt.axis('off')
                plt.imshow(imgs[k])

            plt.show()
            dice = dice_np(np.array([pred]), np.array([truth]))
            print(f'Sample dice is {dice:.4f}')