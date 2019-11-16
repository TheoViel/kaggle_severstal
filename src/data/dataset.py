from util import *
from params import *
from imports import *
from data.masks import *
from fastai.vision import *
from data.transforms import *
from pseudo_labeling import *

def to_softmax(masks):
    mask0 = 1 - np.max(masks, axis=-1)
    return np.concatenate((mask0[:, :, np.newaxis], masks), axis=-1)


def get_classes(image_names, mask_dic):
    classes_max = []
    classes = []
    for img in tqdm(image_names):
        masks = get_masks(img, mask_dic)
        faults = np.sum(masks, axis=(0, 1))
        classes_max.append(np.argmax(faults, axis=0) + 1 if np.sum(masks) else 0)
        classes.append((faults > 0).astype(int))
    return np.array(classes), np.array(classes_max)


class SteelValDataset(Dataset):
    def __init__(self, mask_dic, img_names, classes, transforms):
        super().__init__()
        self.img_names = img_names
        self.mask_dic = mask_dic
        self.transforms = transforms

        self.classes = classes
        self.has_fault = (np.array(self.classes) > 0).astype(int)
        self.black_threshold = 15

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = cv2.imread(TRAIN_IMG_PATH + self.img_names[idx])
        img[img < self.black_threshold] = 0
        img = img / 255
        img = (img - MEAN) / STD

        mask = get_masks(self.img_names[idx], self.mask_dic)
        mask = to_softmax(mask)

        transformed = self.transforms(image=img, mask=mask)
        return transformed['image'], np.argmax(transformed['mask'], axis=0), transformed['mask'][1:, :], self.classes[idx]


class SteelTrainDataset(Dataset):
    def __init__(self, mask_dic, img_names, classes, transforms, kept_imgs=[]):
        super().__init__()

        self.img_names = [img for img in img_names if img in kept_imgs] if len(kept_imgs) else img_names
        self.classes = [classes[i] for i, img in enumerate(img_names) if img in kept_imgs] if len(
            kept_imgs) else classes
        self.has_fault = (np.array(self.classes) > 0).astype(int)
        self.mask_dic = mask_dic
        self.transforms = transforms
        self.black_threshold = 15

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = cv2.imread(TRAIN_IMG_PATH + self.img_names[idx])
        img[img < self.black_threshold] = 0
        img = img / 255

        if np.random.choice(2):
            img = do_random_log_contast(img)
        if not np.random.choice(3):
            img = do_random_noise(img)

        mask = get_masks(self.img_names[idx], self.mask_dic)
        transformed = self.transforms(image=img, mask=mask)

        img = transformed['image']
        mask = transformed['mask']

        if not np.random.choice(3):
            img, mask = add_black_borders_side(img, mask)
        if not np.random.choice(10):
            img, mask = add_black_borders_top(img, mask)

        img = (img - MEAN) / STD
        faults = (np.sum(mask, axis=(0, 1)) > 0).astype(int)
        mask = to_softmax(mask)

        return to_tensor(img), np.argmax(to_tensor(mask), axis=0), faults
#         return to_tensor(img), to_tensor(mask), faults


class SteelTrainDatasetPL(Dataset):
    def __init__(self, mask_dic, img_names, classes, transforms, kept_imgs=[], img_names_pl=PL_IMAGES,
                 mask_dic_pl=MASK_DIC_PL):
        super().__init__()

        self.img_names = [img for img in img_names if img in kept_imgs] if len(kept_imgs) else img_names
        self.img_names_pl = img_names_pl
        self.classes = [classes[i] for i, img in enumerate(img_names) if img in kept_imgs] if len(
            kept_imgs) else classes
        self.has_fault = (np.array(self.classes) > 0).astype(int)
        self.mask_dic = mask_dic
        self.mask_dic_pl = mask_dic_pl
        self.transforms = transforms
        self.black_threshold = 15
        self.pl_idx = len(self.img_names)

    def __len__(self):
        return len(self.img_names) + len(PL_IMAGES)

    def __getitem__(self, idx):
        if idx < self.pl_idx:
            img = cv2.imread(TRAIN_IMG_PATH + self.img_names[idx])
        else:
            img = cv2.imread(TEST_IMG_PATH + self.img_names_pl[idx - self.pl_idx])

        img[img < self.black_threshold] = 0
        img = img / 255

        if np.random.choice(2):
            img = do_random_log_contast(img)
        if not np.random.choice(3):
            img = do_random_noise(img)

        if idx < self.pl_idx:
            mask = get_masks(self.img_names[idx], self.mask_dic)
        else:
            mask = get_masks(self.img_names_pl[idx - self.pl_idx], self.mask_dic_pl)

        transformed = self.transforms(image=img, mask=mask)

        img = transformed['image']
        mask = transformed['mask']

        if not np.random.choice(3):
            img, mask = add_black_borders_side(img, mask)
        if not np.random.choice(10):
            img, mask = add_black_borders_top(img, mask)

        img = (img - MEAN) / STD
        faults = (np.sum(mask, axis=(0, 1)) > 0).astype(int)
        mask = to_softmax(mask)

        return to_tensor(img), np.argmax(to_tensor(mask), axis=0), faults
    
    
class SteelTrainDatasetClf(Dataset):
    def __init__(self, mask_dic, img_names, classes, transforms, kept_imgs=[]):
        super().__init__()
        
        self.img_names = [img for img in img_names if img in kept_imgs] if len(kept_imgs) else img_names
        self.classes = [classes[i] for i, img in enumerate(img_names) if img in kept_imgs] if len(kept_imgs) else classes
        self.has_fault = (np.array(self.classes) > 0).astype(int)
        self.mask_dic = mask_dic
        self.transforms = transforms
        self.black_threshold = 15
        
    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img = cv2.imread(TRAIN_IMG_PATH + self.img_names[idx])
        img[img < self.black_threshold] = 0
        img = img / 255
        
        if np.random.choice(2):
            img = do_random_log_contast(img)
        if not np.random.choice(3): 
            img = do_random_noise(img)
            
        if not np.random.choice(3):
            img = add_black_borders_side(img)
        if not np.random.choice(10):
            img = add_black_borders_top(img)
            
        img = (img - MEAN) / STD

        transformed = self.transforms(image=img)
        return to_tensor(transformed['image']), self.classes[idx]
    
    
class SteelValDatasetClf(Dataset):
    def __init__(self, mask_dic, img_names, classes, transforms):
        super().__init__()
        self.img_names = img_names
        self.mask_dic = mask_dic
        self.transforms = transforms
        self.classes = classes
        self.black_threshold = 15
        
    def __len__(self):
        return len(self.img_names)
    
    
    def __getitem__(self, idx):  
        img = cv2.imread(TRAIN_IMG_PATH + self.img_names[idx])
        img[img < self.black_threshold] = 0
        img = (img / 255 - MEAN) / STD
        return to_tensor(img), self.classes[idx]