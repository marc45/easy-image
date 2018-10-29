#!/usr/bin/python

import numpy as np

from images2npy import folders_name_as_labels


if __name__ == '__main__':
    folders_name_as_labels('images/flowers', 'npy/flowers_batch', batch_size=12,
                           test=0.1, valid=0.1)

    train_x = np.load('npy/flowers_batch/batch_0/train.data.npy')
    train_y = np.load('npy/flowers_batch/batch_0/train.labels.npy')

    print(train_x.shape)
    print(train_y.shape)

