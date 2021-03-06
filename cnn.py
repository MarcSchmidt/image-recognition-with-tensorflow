import logging
import math
import os
import time

import tensorflow as tf

import kubernetes_resolver
import load_images


def create_model(
        input_shape=(32, 32, 3), start_filters=32, kernel_size=(3, 3),
        activation='relu', pool_size=(2, 2), output_classes=10):
    # ---------- Shape the CNN ----------
    #
    # Initialising the CNN as sequential model
    model = tf.keras.models.Sequential()

    # Convolution Layer
    #
    # Transform input in a feature map
    # Conv2D        - Two dimensional input (image)
    # filters       - Amount of filters to use
    # kernel_size   - Shape of each filter kernel
    # padding       - Zero padding ('same' fits input to output shape)
    # activation    - Activation function
    # input_shape   - Shape of input (Rows, Columns, Channels)
    model.add(tf.keras.layers.Conv2D(start_filters, kernel_size, padding='same',
                                     activation=activation,
                                     input_shape=input_shape))
    model.add(tf.keras.layers.Conv2D(start_filters, kernel_size, padding='same',
                                     activation=activation))

    # Pooling Layer
    #
    # Reduce the size of the input data by 75%
    # pool_size - Map 2x2 inputs to 1x1 output
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))

    # Dropout Layer
    # Randomly remove some nodes to add noise and reduce overfitting
    model.add(tf.keras.layers.Dropout(0.25))

    # Convolution Layer
    model.add(tf.keras.layers.Conv2D(start_filters * 2, kernel_size,
                                     padding='same', activation=activation))
    model.add(tf.keras.layers.Conv2D(start_filters * 2, kernel_size,
                                     padding='same', activation=activation))
    # Pooling Layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))

    # Dropout Layer
    model.add(tf.keras.layers.Dropout(0.25))

    # Convolution Layer
    model.add(tf.keras.layers.Conv2D(start_filters * 4, kernel_size,
                                     padding='same', activation=activation))
    model.add(tf.keras.layers.Conv2D(start_filters * 4, kernel_size,
                                     padding='same', activation=activation))
    # Pooling Layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))

    # Dropout Layer
    model.add(tf.keras.layers.Dropout(0.25))

    # Convolution Layer
    model.add(tf.keras.layers.Conv2D(start_filters * 8, kernel_size,
                                     padding='same', activation=activation))
    model.add(tf.keras.layers.Conv2D(start_filters * 8, kernel_size,
                                     padding='same', activation=activation))
    # Pooling Layer
    model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))

    # Dropout Layer
    model.add(tf.keras.layers.Dropout(0.25))

    # Flattening Layer
    # Maps a 3D Matrix to a 1D Vector
    model.add(tf.keras.layers.Flatten())

    # Fully-connected Layer
    # units - Amount of nodes in the hidden layer
    model.add(tf.keras.layers.Dense(units=1024, activation=activation))

    # Dropout Layer
    model.add(tf.keras.layers.Dropout(0.5))

    # Output Layer
    # units - Amount of output classes
    model.add(tf.keras.layers.Dense(units=output_classes, activation='softmax'))

    # Compiling the CNN
    #
    # Adam Optimizer
    optimizer = tf.train.AdamOptimizer()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model


# ---------- Input function ----------
def input_fn(
        img=None,
        label=None,
        batch_size=256,
        num_epochs=None,
        num_workers=3,
        worker_index=None,
        shuffle=True):
    data_set = tf.data.Dataset.from_tensor_slices((img, label))

    if worker_index:
        data_set = data_set.shard(num_workers, worker_index)

    if shuffle:
        data_set = data_set.shuffle(buffer_size=batch_size)

    data_set = data_set.repeat(num_epochs)
    data_set = data_set.batch(batch_size=batch_size)
    return data_set


def model_main():
    start = time.time()
    tf.logging.set_verbosity(tf.logging.DEBUG)
    _logger = logging.getLogger("tensorflow")

    _logger.info("--------------------- Load Kubernetes Config ---------------------")
    tf_config = kubernetes_resolver.build_config()
    os.environ['TF_CONFIG'] = str(tf_config)
    worker_index = kubernetes_resolver.fetch_task_index()
    num_workers = len(kubernetes_resolver.build_worker_list())

    # Local setup
    #
    # worker_index = None
    # num_workers = 3

    _logger.info("--------------------- Load Data ---------------------")
    (x_train, y_train), (x_test, y_test) = load_images.load()

    _logger.info("--------------------- Set RunConfiguration ---------------------")
    distribution = tf.contrib.distribute.CollectiveAllReduceStrategy(num_gpus_per_worker=0)
    config = tf.estimator.RunConfig(train_distribute=distribution,
                                    eval_distribute=distribution)

    # Local setup
    #
    # config = None

    # Create estimator
    _logger.info("--------------------- Create Estimator ---------------------")
    keras_estimator = tf.keras.estimator.model_to_estimator(
        keras_model=create_model(), config=config, model_dir='./model')

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(img=x_train, label=y_train,
                                  num_workers=num_workers,
                                  worker_index=worker_index,
                                  shuffle=True), max_steps=math.floor(1000 / num_workers))
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(img=x_test, label=y_test,
                                  num_workers=num_workers,
                                  worker_index=worker_index,
                                  shuffle=False), steps=100)

    # Create estimator
    tf.LogMessage()
    _logger.info("--------------------- Start Training ---------------------")
    tf.estimator.train_and_evaluate(keras_estimator, train_spec, eval_spec)
    _logger.info("--------------------- Finish training ---------------------")
    end = time.time()
    time_diff = end - start
    _logger.info('--------------------- Estimate time ---------------------')
    _logger.info('Tensorflow Time start: {}'.format(start))
    _logger.info('Tensorflow Time end: {}'.format(end))
    _logger.info('Tensorflow Time elapased: {}'.format(time_diff))
    _logger.info("--------------------- Start Export ---------------------")
    export_dir = keras_estimator.export_savedmodel(
        export_dir_base="./dist",
        serving_input_receiver_fn=serving_input_fn)

    _logger.info("--------------------- Finish Export on Path %s ---------------------"
                 % export_dir)

    _logger.info("--------------------- Start Tensorboard ---------------------")
    if "TF_CONFIG" in os.environ:
        config = os.environ['TF_CONFIG']
        if "\"type\": \"chief\"" in config:
            os.system('tensorboard --logdir=/notebooks/app/model --port=6006')


def serving_input_fn():
    features = {'conv2d_input': tf.placeholder(tf.float32, [None, 32, 32, 3])}
    return tf.estimator.export.ServingInputReceiver(features, features)


# Define the evironment variable, for local usage
# os.environ["TF_CONFIG"] = '{"cluster": ' \
#                           + '{"chief": ["localhost:2223"],' \
#                           + '"worker": ["localhost:2222"]},' \
#                           + '"task": {"type": "chief", "index": 0}}'

# Call the model_main function defined above.
print("Run Tensorflow")
model_main()
