#!/usr/bin/python

from load_data import load_data

from images2npy import files_name_split_as_labels

if __name__ == '__main__':
    files_name_split_as_labels('images/cat&dog', '.', 0, 'npy/cat&dog_batch', batch_size=6,
                               test=0.1, valid=0.1)

    train_x, train_y, valid_x, valid_y = load_data('npy/cat&dog_batch/batch_0')

    print('train data shape', train_x.shape)
    print('train labels shape', train_y.shape)
    print('validation data shape', valid_x.shape)
    print('validation data shape', valid_y.shape)
