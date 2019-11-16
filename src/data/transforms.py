from util import *
from params import *
from imports import *


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def do_random_log_contast(image):
    gain = np.random.uniform(0.70, 1.50, 1)
    if np.random.choice(2, 1):
        image = gain * (2 ** image - 1)
    else:
        image = gain * np.log(image + 1)
    return np.clip(image, 0, 1)


def do_random_noise(image, noise=0.03):
    H, W = image.shape[:2]
    image = image + np.random.uniform(-1, 1, (H, W, 1)) * noise
    return np.clip(image, 0, 1)


def do_random_crop_rescale(image, mask, w, h):
    height, width = image.shape[:2]
    x, y = 0, 0
    if width > w:
        x = np.random.choice(width - w)
    if height > h:
        y = np.random.choice(height - h)
    image = image[y: y + h, x: x + w]
    mask = mask[:, y: y + h, x: x + w]

    if (w, h) != (width, height):
        image = cv2.resize(image, dsize=(width, height),
                           interpolation=cv2.INTER_LINEAR)
        mask = mask.transpose(1, 2, 0)
        mask = cv2.resize(mask, dsize=(width, height),
                          interpolation=cv2.INTER_NEAREST)
        mask = mask.transpose(2, 0, 1)

    return image, mask


def add_black_borders_top(image, mask=None, max_prop=0.3, **kwargs):
    height, width, c = image.shape
    black_height = int(np.random.random() * max_prop * height)

    if np.random.choice(2):
        image[:black_height, :, :] = 0
        if mask is not None:
            mask[:black_height, :, :] = 0
    else:
        image[(height - black_height):, :, :] = 0
        if mask is not None:
            mask[(height - black_height):, :, :] = 0
            
    if mask is not None:
        return image, mask
    else: 
        return image


def add_black_borders_side(image, mask=None, max_prop=0.3, **kwargs):
    height, width, c = image.shape
    black_width = int(np.random.random() * max_prop * width)

    if np.random.choice(2):
        image[:, :black_width, :] = 0
        if mask is not None:
            mask[:, :black_width, :] = 0
    else:
        image[:, (width - black_width):, :] = 0
        if mask is not None:
            mask[:, (width - black_width):, :] = 0
            
    if mask is not None:
        return image, mask
    else: 
        return image



def get_transforms(crop_size=256, test=False):
    if not test:
        if crop_size:
            transforms = albu.Compose([
                albu.RandomCrop(min(IMG_SHAPE[0], crop_size), crop_size),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
            ])
        else:
            transforms = albu.Compose([
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
            ])
    else:
        transforms = albu.Lambda(image=to_tensor, mask=to_tensor)

    return transforms


def get_transforms_resize(crop_size=256, test=False):
    if not test:
        if crop_size:
            transforms = albu.Compose([
                albu.RandomCrop(min(IMG_SHAPE[0], crop_size), crop_size),
                albu.Resize(128, 800, interpolation=cv2.INTER_NEAREST, always_apply=True),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Lambda(image=to_tensor, mask=to_tensor),
            ])
        else:
            transforms = albu.Compose([
                albu.Resize(128, 800, interpolation=cv2.INTER_NEAREST, always_apply=True),
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.Lambda(image=to_tensor, mask=to_tensor),
            ])
    else:
        transforms = albu.Compose([
            albu.Resize(128, 800, interpolation=cv2.INTER_NEAREST, always_apply=True),
            albu.Lambda(image=to_tensor, mask=to_tensor)
        ])
    
    return transforms
