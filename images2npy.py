#!/usr/bin/python

import tensorflow as tf
import numpy as np
import random
import glob
import os
import threading

from tensorflow.python.platform import gfile
from sklearn.model_selection import train_test_split

from process import ProcessBar

IMAGES_EXTENSIONS = [
    'jpg', 'jpeg',
    'png', 'gif', 'bmp'
]

# Block TensorFlow INFO.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def image2array(sess: tf.Session(),
                image_path: str,
                size=None):
    """Convert a image to numpy array.

    :param sess: Tensorflow Session
    :param image_path: Hardware path of the image
    :param size: The size of result array, default [299,299]
    :return: numpy array of the image
    """
    if size is None:
        size = [299, 299]
    image_raw = gfile.FastGFile(image_path, 'rb').read()
    image_type = image_path.split('/')[-1].split('\\')[-1].split('.')[-1].lower()
    if image_type == 'jpg' or image_type == 'jpeg':
        image_decoded = tf.image.decode_jpeg(image_raw)
    elif image_type == 'png':
        image_decoded = tf.image.decode_png(image_raw)
    elif image_type == 'bmp':
        image_decoded = tf.image.decode_bmp(image_raw)
    elif image_type == 'gif':
        image_decoded = tf.image.decode_gif(image_raw)
    else:
        raise ValueError('File {} is not a known image'.format(image_path))

    # Type must be float32
    if image_decoded.dtype != tf.float32:
        image_decoded = tf.image.convert_image_dtype(
            image_decoded, dtype=tf.float32)

    image_decoded = tf.image.resize_images(image_decoded, size=size)

    return sess.run(image_decoded)


def choose_file(all_files: list,
                labels: list,
                size: int):
    """
    Select a certain number of elements from the array,
    delete them in the original array and return them.

    :param all_files: current files
    :param labels: current labels
    :param size: Chosen files' size
    :return: Chosen files.
    """
    if size >= len(all_files):
        chosen = all_files.copy()
        chosen_labels = labels.copy()
        all_files.clear()
        return chosen, chosen_labels

    chosen_files = random.sample(all_files, size)
    chosen_labels = []

    for chosen in chosen_files:
        index = all_files.index(chosen)
        chosen_labels.append(labels[index])
        del all_files[index]
        del labels[index]

    return chosen_files, chosen_labels


def get_all_files(path: str):
    """Get all the image files in a directory and return a list of their paths

    :param path: Target path
    :return: List, element are the path of image file.
    """
    global IMAGES_EXTENSIONS
    file_list = []
    for ext in IMAGES_EXTENSIONS:
        file_glob = os.path.join(path, '*.' + ext)
        file_list.extend(glob.glob(file_glob))
    return file_list


def paths2npy(sess: tf.Session(),
              paths: list,
              labels: list,
              save_path: str,
              load_type: str,
              size=None):
    """Convert images from multiple paths directly into an array and store them

    :param load_type: Train or Test or Validation
    :param sess: TensorFlow Session
    :param paths: Paths stored images
    :param labels: Paths' labels
    :param save_path: path to save npy file
    :param size: The size of result array, default [299,299]
    """
    print('Begin to handle %s data' % load_type)
    bar = ProcessBar(len(paths), done_msg='Handled %d images.' % len(paths))
    data = []
    for path in paths:
        images_data = image2array(sess, path, size=size)
        data.append(images_data)
        bar.show_process()
    np.save(save_path + '.data.npy', np.array(data))
    np.save(save_path + '.labels.npy', np.array(labels))
    print('Saved data at', save_path.replace('\\', '/'))
    del data


def images2npy(paths: list,
               labels: list,
               save_path: str,
               test=0.3,
               valid=0.3,
               size=None):
    """Convert multiple images to a numpy array and save them as npy files

    :param paths: Path to all pending images
    :param labels: Path's corresponding label
    :param save_path: The npy path to save
    :param test: The proportion of Test data in all data
    :param valid: The proportion of Validation data in Train data
    :param size: The size of result array, default [299,299]
    """
    sess = tf.Session()
    train_paths, test_paths, train_labels, test_labels = \
        train_test_split(paths, labels, test_size=test, random_state=66)
    train_paths, valid_paths, train_labels, valid_labels = \
        train_test_split(train_paths, train_labels, test_size=valid, random_state=77)
    print('Handling Task: %d training data, %d testing data, %d validation data.'
          % (len(train_labels), len(test_labels), len(valid_labels)))

    paths2npy(sess, train_paths, train_labels, os.path.join(save_path, 'train'), 'Train', size)
    paths2npy(sess, test_paths, test_labels, os.path.join(save_path, 'test'), 'Test', size)
    paths2npy(sess, valid_paths, valid_labels, os.path.join(save_path, 'valid'), 'Validation', size)
    sess.close()


class BatchThread(threading.Thread):
    """
    Thread that processes a batch of data.
    Processing each batch of data starts a new thread and session.
    """
    def __init__(self, files, labels, save_path, test, valid, size):
        threading.Thread.__init__(self)
        self.file = files
        self.labels = labels
        self.save_path = save_path
        self.test = test
        self.valid = valid
        self.size = size

    def run(self):
        images2npy(self.file, self.labels, self.save_path,
                   self.test, self.valid, self.size)


def batch_handle(image_files: list,
                 labels: list,
                 save_path: str,
                 test: float,
                 valid: float,
                 image_size=None,
                 batch_size=None):
    """
    Receive the file list, if it need to process the data in batches, implement batching.
    If it don't need batching, pass the file list to BatchThread for processing.

    :param image_files: All image file names' list
    :param labels: All image files' labels
    :param save_path: path to save npy files.
    :param test: Test rate
    :param valid: Validation rate
    :param image_size: The size of result array, default [299,299]
    :param batch_size: The size of batch, None if do not use batch.(default)
    """
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    print('=' * 70)
    if batch_size is None:
        print('No Batch, all images will be handled at once.')
        images2npy(image_files, labels,
                   save_path, test, valid, image_size)
    else:

        if type(batch_size) != int:
            raise TypeError('batch_size must be an integer')

        arr_len = len(image_files)
        if arr_len % batch_size == 0:
            total_batch = int(arr_len / batch_size)
        else:
            total_batch = int(arr_len // batch_size + 1)

        print('Batch Task, batch size={}, data length={}, get {}'
              ' batches to handle'.format(batch_size, arr_len, total_batch))
        batch = 0
        while len(image_files) > 0:
            print('=' * 70)
            print('Batch {}/{}'.format(batch, total_batch - 1))
            batch_files, batch_labels = choose_file(image_files, labels, batch_size)
            batch_path = os.path.join(save_path, 'batch_%d' % batch)

            if not os.path.isdir(batch_path):
                os.mkdir(batch_path)

            print('Directory: %s' % batch_path.replace('\\', '/'))
            print('=' * 70)
            thread = BatchThread(batch_files, batch_labels, batch_path, test, valid, image_size)
            thread.start()
            thread.join()
            batch += 1


def folders_name_as_labels(path: str,
                           save_path: str,
                           batch_size=None,
                           test=0.3,
                           valid=0.3,
                           image_size=None):
    """Read all directories in a directory, each directory holds all the images of a tag.

    :param path: target directory
    :param save_path: path to save npy files.
    :param test: Test rate
    :param valid: Validation rate
    :param image_size: The size of result array, default [299,299]
    :param batch_size: The size of batch, None if do not use batch.(default)
    """
    sub_dirs = [d[0] for d in os.walk(path)]
    is_root = True

    image_files = []
    labels = []

    label = 0
    for sub_dir in sub_dirs:
        if is_root:
            is_root = False
            continue
        sub_image_files = get_all_files(sub_dir)
        image_files.extend(sub_image_files)
        labels.extend([label] * len(sub_image_files))
        label += 1
    batch_handle(image_files, labels, save_path,
                 test, valid, image_size, batch_size)


def files_name_split_as_labels(path: str,
                               split_char: str,
                               label_index,
                               save_path: str,
                               batch_size=None,
                               test=0.3,
                               valid=0.3,
                               image_size=None):
    """
    Read all the pictures in a directory, with the file name as the label.
    Among them, you need to provide the separator and subscript.
    The file name will be split using the separator, and then the split subscript will be selected as the label.

    :param image_size: The size of result array, default [299,299]
    :param save_path: path to save npy files.
    :param path: target directory
    :param split_char: The character used to split file name.
    :param label_index: The index of label after split.
    :param batch_size: The size of batch, None if do not use batch.(default)
    :param test: Test rate
    :param valid: Validation rate
    """
    images_files = get_all_files(path)
    label_vocab = {}
    labels = []
    label = 0
    for image_file in images_files:
        file_name = image_file.split('/')[-1].split('\\')[-1]
        label_char = file_name.split(split_char)[label_index]
        if label_char not in label_vocab:
            label_vocab[label_char] = label
            labels.append(label)
            label += 1
        else:
            labels.append(label_vocab[label_char])
    batch_handle(images_files, labels, save_path, test=test, valid=valid, batch_size=batch_size,
                 image_size=image_size)
