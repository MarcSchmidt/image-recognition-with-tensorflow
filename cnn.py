# ---------- Imports ----------
import json
import os
from time import time

from tensorflow import estimator as tf_estimator
from tensorflow.contrib.distribute import CollectiveAllReduceStrategy

import tensorflow as tf

from tensorflow import keras as ks
import numpy as np

# ---------- Shape the CNN ----------
# Initialising the CNN as sequential model
classifier = ks.models.Sequential()

# Convolution Layer
# Transform input in a feature map
# Conv2D        - Two dimensional input (Image) with 4 Parameter
# 32            - Amount of Filters to use
# (3, 3)        - shape of each Filter
# (64, 64, 3)   - Shape of Input (Rows, Columns, Channels)
# 'relu'        - Activation function
classifier.add(ks.layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(ks.layers.Conv2D(32, (3, 3), activation='relu'))

# Pooling Layer
# Reduce the size of the input data by 75%
# (2, 2) - Map 2x2 inputs to one output
classifier.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
# Dropout Layer
# Remove randomly some nodes to add noise
classifier.add(ks.layers.Dropout(0.25))

# Convolution Layer
classifier.add(ks.layers.Conv2D(64, (3, 3), activation='relu'))
classifier.add(ks.layers.Conv2D(64, (3, 3), activation='relu'))
classifier.add(ks.layers.MaxPooling2D(pool_size=(2, 2)))
classifier.add(ks.layers.Dropout(0.25))

# Flattening Layer
# Maps a X by X Matrix to one Vector
classifier.add(ks.layers.Flatten())

# Full connection Layer
# units  - Amound of nodes in the hidden layer
classifier.add(ks.layers.Dense(units=512, activation='relu'))
classifier.add(ks.layers.Dropout(0.5))

# Output Layer
# units  - Amount of output classes
classifier.add(ks.layers.Dense(units=10, activation='sigmoid'))

# Compiling the CNN
optimizer = tf.train.AdamOptimizer()
classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

classifier.summary()


# ---------- Load Data ----------


def training_input_fn():
    # Generator to create batches of data from inputs
    # rescale           - Rescaling factors, it means that the RGB input with range 0-255 will be mapped to 0-1 values
    # rotation_range    - randomly rotate input by a degree
    # shear_range       - randomly apply shearing transformation
    # zoom_range        - randomly zooming into the input
    # horizontal_flip   - randomly flip the input horizontally
    train_datagen = ks.preprocessing.image.ImageDataGenerator(rescale=1. / 255,
                                                              rotation_range=0,
                                                              shear_range=0.2,
                                                              zoom_range=0.2,
                                                              horizontal_flip=True)
    # Load the training data from directory, where each subdirectory is one class
    # target_size   - resize images to the given size
    # batch_size    - Amount of Images to load for one Step in training (should fit into system memory)
    # class_mode    - Set classes to 2D one-hot encoded labels
    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='categorical')

    return tf.data.Dataset.from_generator(training_set, output_types=(tf.int64, tf.int64))


def test_input_fn():
    # Generate test data
    test_datagen = ks.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    # Load the test data from directory, where each subdirectory is one class
    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='categorical')
    return tf.data.Dataset.from_generator(test_set, output_types=(tf.int64, tf.int64))


def input_fn():
    x = np.random.random((1024, 10))
    y = np.random.randint(2, size=(1024, 1))
    x = tf.cast(x, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(32)
    return dataset


# ---------- Start Training ----------
# Configure Tensorboard for Keras with defined log name
tensorboard = ks.callbacks.TensorBoard(log_dir="logs/{}".format(time()),
                                       histogram_freq=0,
                                       write_graph=True, write_images=True)

# Train the cnn with given training_set
# steps_per_epoch   -  50.000 Images in total which are split in Batches of 32 Images makes 1562,5 Steps
# epochs            - One epoch equal training one time on the whole dataset (or all steps)
# validation_steps  - 10.000 Images total which are split in Batches of 32 Images makes 312 Steps
# callback          - gives information of training to the tensorboard
# training_set = training_input_fn()
# test_set = test_input_fn()
# classifier.fit_generator(training_set,
#                          steps_per_epoch=1562,
#                          epochs=25,
#                          validation_data=test_set,
#                          validation_steps=312,
#                          callbacks=[tensorboard])

(train_img, train_label), (test_img, test_label) = tf.keras.datasets.cifar10.load_data()


def input_training():
    # x = tf.cast(input_train_data_img, tf.float32)
    # training_set = tf.data.Dataset.from_tensor_slices((x, input_train_data_label))
    training_set = tf.data.Dataset.from_tensor_slices((train_img, train_label))
    training_set.batch(32)

    return training_set


def input_validation():
    # input_eval_data_img, input_eval_data_label = input_train_data_img, input_train_data_label
    # x = tf.cast(input_eval_data_img, tf.float32)
    # validation_set = tf.data.Dataset.from_tensor_slices((x, input_eval_data_label))
    validation_set = tf.data.Dataset.from_tensor_slices((test_img, test_label))
    validation_set.batch(32)
    return validation_set


def model_main():
    distribution = CollectiveAllReduceStrategy(num_gpus_per_worker=1)
    run_config = tf_estimator.RunConfig(train_distribute=distribution, eval_distribute=distribution)

    # Create estimator
    keras_estimator = ks.estimator.model_to_estimator(
        keras_model=classifier, config=run_config, model_dir='./model')

    train_spec = tf_estimator.TrainSpec(input_fn=input_training)
    eval_spec = tf_estimator.EvalSpec(input_fn=input_validation)
    tf_estimator.train_and_evaluate(keras_estimator, train_spec, eval_spec)


# Call the model_main function defined above.
model_main()
