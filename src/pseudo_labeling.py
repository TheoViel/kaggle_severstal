import numpy as np
import pandas as pd


try:
    THRESHOLD_CONFIDENT_FAULT = 0.4
    THRESHOLD_CONFIDENT_NO_FAULT = 0.2

    IMG_TEST = pd.read_csv('../input/sample_submission.csv')['ImageId_ClassId'].apply(lambda x: x[:-2]).values[::4]

    pl_probs = np.load('../output/all_probs_test.npy')
    PL_LABELS = pl_probs

    pl_df = pd.read_csv('../output/pl_9195.csv').fillna('')
    pl_df['EncodedPixels2'] = pd.read_csv('../output/pl_9193.csv').fillna('')['EncodedPixels']

    pl_df['probs'] = PL_LABELS.flatten()

    pl_df['faulty'] = pl_df['EncodedPixels'].apply(lambda x: int(len(x) > 0))
    pl_df['faulty2'] = pl_df['EncodedPixels2'].apply(lambda x: int(len(x) > 0))

    pl_df['kept_pos'] = np.min((pl_df['probs'] > THRESHOLD_CONFIDENT_FAULT, pl_df['faulty'], pl_df['faulty2']), axis=0)
    pl_df['kept_neg'] = np.min((pl_df['probs'] < THRESHOLD_CONFIDENT_NO_FAULT, 1 - pl_df['faulty'], 1 - pl_df['faulty2']), axis=0)
    pl_df['kept'] = pl_df['kept_pos'] + pl_df['kept_neg'] > 0

    kept_imgs = np.min(pl_df['kept'].values.reshape(-1, 4), axis=1)
    pl_df['kept_all'] = np.repeat(kept_imgs, 4)

    kept_classes = pl_df[pl_df['kept_all']]['faulty']
    print(f'Kept {kept_classes.shape[0] / 4 :.0f} images out of {len(IMG_TEST)}')
    for i in range(4):
        print(f'Number of defects of class {i} :', np.sum(kept_classes[i::4]))

    PL_DF = pl_df[pl_df['kept_all']][['ImageId_ClassId', 'EncodedPixels', 'faulty']]

    PL_DF['ImageId'] = PL_DF['ImageId_ClassId'].apply(lambda x: x.split('_')[0])
    PL_DF['ClassId'] = PL_DF['ImageId_ClassId'].apply(lambda x: x.split('_')[1])

    group_img_pl = PL_DF[['ImageId', 'EncodedPixels']].groupby('ImageId').agg(list)
    rep_classes_pl = group_img_pl['EncodedPixels'].apply(pd.Series).rename(columns=lambda x : str(x))
    rep_classes_pl['ClassNumber'] = group_img_pl['EncodedPixels'].apply(lambda x: len([i for i in x if i != ""]))

    PL_IMAGES = rep_classes_pl.index.values
    MASK_DIC_PL = rep_classes_pl.drop('ClassNumber', axis=1).to_dict('index')
except:
    PL_IMAGES = []
    MASK_DIC_PL = []
