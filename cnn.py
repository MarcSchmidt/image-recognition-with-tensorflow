# ---------- Imports ----------
import os

import numpy as np
import tensorflow as tf
from tensorflow import estimator as tf_estimator
from tensorflow import keras as ks
from tensorflow.contrib.distribute import CollectiveAllReduceStrategy


def create_model(input_shape=(32, 32, 3), starting_filter_size=32, filter_shape=(3, 3), activation_fn='relu',
                 pooling_shape=(2, 2), output_classes=10):
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
    classifier.add(
        ks.layers.Conv2D(starting_filter_size, filter_shape, input_shape=input_shape, activation=activation_fn,
                         data_format='channels_last'))
    classifier.add(ks.layers.Conv2D(starting_filter_size, filter_shape, activation=activation_fn))

    # Pooling Layer
    # Reduce the size of the input data by 75%
    # (2, 2) - Map 2x2 inputs to one output
    classifier.add(ks.layers.MaxPooling2D(pool_size=pooling_shape))
    # Dropout Layer
    # Remove randomly some nodes to add noise
    classifier.add(ks.layers.Dropout(0.25))

    # Convolution Layer
    classifier.add(ks.layers.Conv2D(starting_filter_size * 2, filter_shape, activation=activation_fn))
    classifier.add(ks.layers.Conv2D(starting_filter_size * 2, filter_shape, activation=activation_fn))
    classifier.add(ks.layers.MaxPooling2D(pool_size=pooling_shape))
    classifier.add(ks.layers.Dropout(0.25))

    # Flattening Layer
    # Maps a X by X Matrix to one Vector
    classifier.add(ks.layers.Flatten())

    # Full connection Layer
    # units  - Amound of nodes in the hidden layer
    classifier.add(ks.layers.Dense(units=512, activation=activation_fn))
    classifier.add(ks.layers.Dropout(0.5))

    # Output Layer
    # units  - Amount of output classes
    classifier.add(ks.layers.Dense(units=output_classes, activation='sigmoid'))

    # Compiling the CNN
    optimizer = tf.train.AdamOptimizer()
    classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    classifier.summary()

    return classifier


# ---------- Input function ----------
def input_fn(img=None,
             label=None,
             batch_size=32,
             num_epochs=None,
             num_worker=3,
             shuffle=True):
    data_set = tf.data.Dataset.from_tensor_slices((img, label))

    if shuffle:
        data_set = data_set.shuffle(buffer_size=batch_size * num_worker)

    data_set = data_set.repeat(num_epochs)
    data_set = data_set.batch(batch_size=batch_size)
    return data_set


def model_main():
    print("--------------------- Load Data ---------------------")
    (train_img, train_label), (test_img, test_label) = tf.keras.datasets.cifar10.load_data()

    # Reduce data to 10% to not exceed the given memory
    train_img = np.array_split(train_img, 10)[0]
    train_label = np.array_split(train_label, 10)[0]
    test_img = np.array_split(test_img, 10)[0]
    test_label = np.array_split(test_label, 10)[0]

    # Convert class vectors to binary class matrices.
    y_train = ks.utils.to_categorical(train_label, 10)
    y_test = ks.utils.to_categorical(test_label, 10)

    train_img = train_img.astype('float32')
    test_img = test_img.astype('float32')

    # Map RGB values from 0-255 to 0-1
    train_img /= 255
    test_img /= 255

    print("--------------------- Set RunConfiguration ---------------------")
    distribution = CollectiveAllReduceStrategy(num_gpus_per_worker=1)
    run_config = tf_estimator.RunConfig(train_distribute=distribution, eval_distribute=distribution)

    # Create estimator
    print("--------------------- Create Estimator ---------------------")
    keras_estimator = ks.estimator.model_to_estimator(
        keras_model=create_model(), config=run_config, model_dir='./model')

    train_spec = tf_estimator.TrainSpec(input_fn=lambda: input_fn(img=train_img, label=y_train, shuffle=True),
                                        max_steps=1000)
    eval_spec = tf_estimator.EvalSpec(input_fn=lambda: input_fn(img=test_img, label=y_test, shuffle=False),
                                      steps=100)

    # Create estimator
    print("--------------------- Start Training ---------------------")
    tf_estimator.train_and_evaluate(keras_estimator, train_spec, eval_spec)

    print("--------------------- Finish training ---------------------")

    if "TF_CONFIG" in os.environ:
        config = os.environ['TF_CONFIG']
        if "\"type\": \"chief\"" in config:
            os.system('tensorboard --logdir=/notebooks/app/model')


# Define the evironment variable, for local usage
# os.environ[
#    "TF_CONFIG"] = '{"cluster": {"chief": ["localhost:2223"],"worker": ["localhost:2222"]},"task": {"type": "chief", "index": 0}}'

# Call the model_main function defined above.
#
model_main()
