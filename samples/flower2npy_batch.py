#!/usr/bin/python

import numpy as np

from images2npy import folders_name_as_labels


if __name__ == '__main__':
    folders_name_as_labels('images/flowers', 'npy/flowers_batch', batch_size=12,
                           test=0.1, valid=0.1)

    train_sample_data = np.load('npy/flowers_batch/batch_0/train.npy')
    valid_sample_data = np.load('npy/flowers_batch/batch_0/valid.npy')

    train_x, train_y = train_sample_data[0], train_sample_data[1]
    print(train_y)
