import os
import pathlib as pl

import numpy as np
import tensorflow as tf


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def load():
    train_dir = pl.Path('dataset/training_set')
    test_dir = pl.Path('dataset/test_set')

    # for later use to map back label ids to label names
    label_dict = {}

    x_train = []
    y_train = []

    for i, cur_dir in enumerate(sorted(train_dir.iterdir())):
        if cur_dir.is_dir():
            label = os.path.basename(str(cur_dir))
            label_dict[i] = label
            for img in cur_dir.iterdir():
                x_train.append(tf.keras.preprocessing.image.img_to_array(
                        tf.keras.preprocessing.image.load_img(img)))
                y_train.append(i)

    x_train = np.array(x_train) / 255.0
    y_train = np.array(y_train)
    y_train = y_train.reshape(y_train.shape + (1,))

    x_test = []
    y_test = []

    for i, cur_dir in enumerate(sorted(test_dir.iterdir())):
        if cur_dir.is_dir():
            for img in cur_dir.iterdir():
                x_test.append(tf.keras.preprocessing.image.img_to_array(
                        tf.keras.preprocessing.image.load_img(img)))
                y_test.append(i)

    x_test = np.array(x_test) / 255.0
    y_test = np.array(y_test)
    y_test = y_test.reshape(y_test.shape + (1,))

    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    x_test, y_test = unison_shuffled_copies(x_test, y_test)

    # Reduce data to 10% to not exceed the given memory
    # x_train = np.array_split(x_train, 10)[0]
    # y_train = np.array_split(y_train, 10)[0]
    # x_test = np.array_split(x_test, 10)[0]
    # y_test = np.array_split(y_test, 10)[0]

    # Convert class vectors to binary class matrices.
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    return (x_train, y_train), (x_test, y_test)
