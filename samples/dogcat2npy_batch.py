#!/usr/bin/python

import numpy as np

from images2npy import files_name_split_as_labels

if __name__ == '__main__':
    files_name_split_as_labels('images/cat&dog', '.', 0, 'npy/cat&dog_batch', batch_size=6,
                               test=0.1, valid=0.1)

    train_x = np.load('npy/cat&dog_batch/batch_0/train.data.npy')
    train_y = np.load('npy/cat&dog_batch/batch_0/train.labels.npy')

    print(train_x.shape)
    print(train_y.shape)
