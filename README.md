# Kaggle : Severstal Steel Defect Detection  [WIP]

## On the competition 

See https://www.kaggle.com/c/severstal-steel-defect-detection

The goal of this competition is to detect faults on steel plates. There are 4 kinds of defect to predict, and their repartition is unbalanced. 

> The production process of flat sheet steel is especially delicate. From heating and rolling, to drying and cutting, several machines touch flat steel by the time it’s ready to ship. Today, Severstal uses images from high frequency cameras to power a defect detection algorithm. In this competition, you’ll help engineers improve the algorithm by localizing and classifying surface defects on a steel sheet.

In addition, inference had to be made in less than one hour, which prevented extensive model stacking.

The competition took place from July 25 2019 to October 25 2019

## Metric

See https://www.kaggle.com/c/severstal-steel-defect-detection
/overview/evaluation

This competition is evaluated on the mean Dice coefficient, which measures how well predicted masks correspond to the ground truth :

> ![equation](https://latex.codecogs.com/gif.latex?%5Cfrac%7B2%20*%20%7CX%20%5Ccap%20Y%7C%7D%7B%7CX%7C%20&plus;%20%7CY%7C%7D)

For a given image, its score is the average dice on the four classes. 
Masks with no defects for a class score 0 if a defect is predicted, and 1 otherwise. False positive are heavily penalized.

## Results

I teamed up with Maxim Shugaev and Miguel Pinto in this competiton. After reaching the 19th place on the public leaderboard, we ended up **40th** overall.
The discrepancy between training data and first stage data was huge, and the same happened with 2nd stage data, which resulted in many teams dropping down.

Our metric results were *0.91950* on public and *0.90354* on private. 

## Solution Overview

Our solution is a two step pipeline : 
- First remove images with no faults with a classifier
- Segment the remaining images


### Models

#### Classification

 - Simple Resnet34, we used two folds out of 5.
 - Test Time Augmentation : None , hflip, vflip, both

#### Segmentation

- We went for a voting ensemble of 7 models trained on the first of our 5 folds:
    - UnetSeResNext50 x 4
    - UnetDensenet169
    - FPNEfficientnet x 2

SeResNexts performed better but we still wanted to add some diversity.

- Test Time Augmentation : None, hflip, vflip

### Tricks for segmentation
- Auxiliary classifier head (4 classes)
- Lovasz Loss (5 classes + softmax) + BCE for the classification.
- Training on progressive crop size : 256x256 -> 512x256 -> 1024x256 -> Full resolution
- Augments :
    - Crops
    - Flips
    - Brightness
    - Putting black bars on one of the sides to simulate the edge of the plate

## Repository 

- `input` : Input data is expected here
- `src` : Code
- `notebooks` : Notebooks

## Data

Data can be downloaded on the official Kaggle page : https://www.kaggle.com/c/severstal-steel-defect-detection/data

## Ressources

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587.pdf)
- [The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks](https://arxiv.org/pdf/1705.08790.pdf)

## Results preview
