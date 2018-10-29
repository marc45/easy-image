#!/usr/bin/python

from images2npy import folders_name_as_labels
from load_data import load_data


if __name__ == '__main__':
    folders_name_as_labels('images/flowers', 'npy/flowers_batch', batch_size=12,
                           test=0.1, valid=0.1)

    train_x, train_y, valid_x, valid_y = load_data('npy/flowers_batch/batch_0')

    print('train data shape', train_x.shape)
    print('train labels shape', train_y.shape)
    print('validation data shape', valid_x.shape)
    print('validation data shape', valid_y.shape)

