from data.dataset import *
from training.train import *
from training.freezing import *
from data.transforms import get_transforms as transfo

from model_zoo.unet import *

seed_everything(seed)

# CP_PATH = f'checkpoints/{date.today()}/'
# if not os.path.exists(CP_PATH):
#     os.mkdir(CP_PATH)

today = date.today()
print("Today's date:", today)

df_train = pd.read_csv(DATA_PATH + 'train.csv')
df_train['EncodedPixels'].fillna('-1', inplace=True)
print('Number of training images : ', len(df_train) // 4)

df_train['ImageId'] = df_train['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
df_train['ClassId'] = df_train['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

group_img = df_train[['ImageId', 'EncodedPixels']].groupby('ImageId').agg(list)
rep_classes = group_img['EncodedPixels'].apply(pd.Series).rename(columns=lambda x : str(x))
rep_classes['ClassNumber'] = group_img['EncodedPixels'].apply(lambda x: len([i for i in x if i != "-1"]))

all_images = rep_classes.index.values
hard_negatives = rep_classes[rep_classes['ClassNumber'] == 0].index.values
positives = rep_classes[rep_classes['ClassNumber'] > 0].index.values

print('Number of images with defaults: ', len(positives))

mask_dic = rep_classes.drop('ClassNumber', axis=1).to_dict('index')

classes, classes_max = get_classes(all_images, mask_dic)


def k_fold_training(create_model, backbone, images, classes, classes_max, mask_dic, transforms_dic,
                    kept_for_training=([],), k=5, selected_fold=0, use_aux_clf=False,
                    batch_sizes=(32,), epochs=(5,), seed=2019, lr=(1e-3,), min_lrs=(1e-5,),
                    verbose=1, save=True, cp=False, warmup_props=(0.1,), model_name="model", pretrained_path=''):
    splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(images, classes_max))
    train_idx, val_idx = splits[selected_fold]
    i = selected_fold

    print(f"-------------   Fold {i + 1}  -------------")
    seed_everything(seed + i)

    if backbone in SETTINGS.keys():
        model = create_model(SETTINGS[backbone], num_classes=4, center_block="aspp", aux_clf=use_aux_clf)
    else:
        model = create_model(num_classes=4, center_block="aspp", aux_clf=use_aux_clf)

    if len(pretrained_path):
        load_model_weights(model, pretrained_path)

    val_dataset = SteelValDataset(mask_dic, images[val_idx], classes[val_idx], transforms_dic["val"])

    freeze_encoder(model)
    n_parameters = count_parameters(model)

    print(f'\n - Training with 256x256 crops - Frozen encoder: {n_parameters} trainable parameters\n')

    train_dataset = SteelTrainDataset(mask_dic, images[train_idx], classes[train_idx], transforms_dic["train"][0],
                                      kept_imgs=kept_for_training[0])

    fit_seg(model, train_dataset, val_dataset, epochs=epochs[0], batch_size=batch_sizes[0],
            lr=lr[0], min_lr=min_lrs[0], schedule='cosine', use_aux_clf=use_aux_clf,
            warmup_prop=warmup_props[0], verbose=verbose)

    if cp:
        load_model_weights(model, f"{model_name}_{i + 1}_0.pt", verbose=1)
    elif save:
        save_model_weights(model, f"{model_name}_{i + 1}_0.pt", verbose=1)

    unfreeze_encoder(model)
    n_parameters = count_parameters(model)

    print(f'\n - Training with 256x256 crops - {n_parameters} trainable parameters\n')

    train_dataset = SteelTrainDataset(mask_dic, images[train_idx], classes[train_idx], transforms_dic["train"][1],
                                      kept_imgs=kept_for_training[1])

    fit_seg(model, train_dataset, val_dataset, epochs=epochs[1], batch_size=batch_sizes[1],
            lr=lr[1], min_lr=min_lrs[1], schedule='cosine', use_aux_clf=use_aux_clf,
            warmup_prop=warmup_props[1], verbose=verbose)

    if cp:
        load_model_weights(model, f"{model_name}_{i + 1}_1.pt", verbose=1)
    elif save:
        save_model_weights(model, f"{model_name}_{i + 1}_1.pt", verbose=1)

    print('\n - Training with 512x256 crops \n')

    train_dataset = SteelTrainDatasetPL(mask_dic, images[train_idx], classes[train_idx], transforms_dic["train"][2],
                                        kept_imgs=kept_for_training[2])

    fit_seg(model, train_dataset, val_dataset, epochs=epochs[2], batch_size=batch_sizes[2],
            lr=lr[2], min_lr=min_lrs[2], schedule='cosine', use_aux_clf=use_aux_clf,
            warmup_prop=warmup_props[2], verbose=verbose)

    if cp:
        load_model_weights(model, f"{model_name}_{i + 1}_2.pt", verbose=1)
    elif save:
        save_model_weights(model, f"{model_name}_{i + 1}_2.pt", verbose=1)

    print('\n - Training with 1024x256 crops \n')

    train_dataset = SteelTrainDataset(mask_dic, images[train_idx], classes[train_idx], transforms_dic["train"][3],
                                      kept_imgs=kept_for_training[3])

    fit_seg(model, train_dataset, val_dataset, epochs=epochs[3], batch_size=batch_sizes[3],
            lr=lr[3], min_lr=min_lrs[3], schedule='cosine', use_aux_clf=use_aux_clf,
            warmup_prop=warmup_props[3], verbose=verbose)

    if cp:
        load_model_weights(model, f"{model_name}_{i + 1}_3.pt", verbose=1)
    elif save:
        save_model_weights(model, f"{model_name}_{i + 1}_3.pt", verbose=1)

    print('\n - Training with full images \n')

    train_dataset = SteelTrainDataset(mask_dic, images[train_idx], classes[train_idx], transforms_dic["train"][4],
                                      kept_imgs=kept_for_training[4])

    fit_seg(model, train_dataset, val_dataset, epochs=epochs[4], batch_size=batch_sizes[4],
            lr=lr[4], min_lr=min_lrs[4], schedule='cosine', use_aux_clf=use_aux_clf,
            warmup_prop=warmup_props[4], verbose=verbose)

    if cp:
        load_model_weights(model, f"{model_name}_{i + 1}_4.pt", verbose=1)
    elif save:
        save_model_weights(model, f"{model_name}_{i + 1}_4.pt", verbose=1)

    print('\n - Training with 1024x256 crops and pseudo-labels\n')

    train_dataset = SteelTrainDatasetPL(mask_dic, images[train_idx], classes[train_idx], transforms_dic["train"][5],
                                        kept_imgs=kept_for_training[5])

    fit_seg(model, train_dataset, val_dataset, epochs=epochs[5], batch_size=batch_sizes[5],
            lr=lr[5], min_lr=min_lrs[5], schedule='cosine', use_aux_clf=use_aux_clf,
            warmup_prop=warmup_props[5], verbose=verbose)

    if cp:
        load_model_weights(model, f"{model_name}_{i + 1}_5.pt", verbose=1)
    elif save:
        save_model_weights(model, f"{model_name}_{i + 1}_5.pt", verbose=1)


backbone = 'resnet34'
build_model = SegmentationUnet
model_name = "unet_" + backbone


kept_images = [[]]*7

k = 5
selected_fold = 0


transforms_dic = {"train": [transfo(crop_size=256),
                            transfo(crop_size=256),
                            transfo(crop_size=512),
                            transfo(crop_size=1024),
                            transfo(crop_size=0),
                            transfo(crop_size=1024),
                            ],
                  "val": transfo(test=True)}

# batch_sizes = [32, 32, 16, 8, 4] # resnet
batch_sizes = [16, 16, 8, 4, 2, 4]
epochs = [5, 30, 30, 20, 15, 20]
# epochs = [1] * 7

warmup_props = [0, 0.1, 0.1, 0.1, 0.1, 0.2]
lrs = [1e-3, 1e-3, 5e-4, 5e-5, 1e-5, 5e-5]

min_lrs = [1e-3, 1e-4, 5e-6, 1e-6, 1e-6, 1e-6]

tta = True
use_aux_clf = True

assert torch.cuda.is_available(), 'Training on GPU is mandatory'

k_fold_training(build_model, backbone, all_images, classes, classes_max, mask_dic, transforms_dic,
                kept_images, k=k, selected_fold=selected_fold, use_aux_clf=use_aux_clf,
                batch_sizes=batch_sizes, epochs=epochs, warmup_props=warmup_props, lr=lrs, min_lrs=min_lrs,
                verbose=1, save=True, cp=False, model_name=model_name, seed=seed)

print('Training Done.')
