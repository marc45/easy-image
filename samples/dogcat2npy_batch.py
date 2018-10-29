#!/usr/bin/python

import numpy as np

from images2npy import files_name_split_as_labels

if __name__ == '__main__':
    files_name_split_as_labels('images/cat&dog', '.', 0, 'npy/cat&dog_batch', batch_size=6,
                               test=0.1, valid=0.1)

    train_sample_data = np.load('npy/cat&dog_batch/batch_0/train.npy')
    train_valid_data = np.load('npy/cat&dog_batch/batch_0/valid.npy')

    train_x, train_y = train_sample_data[0], train_sample_data[1]
    print(train_x)
