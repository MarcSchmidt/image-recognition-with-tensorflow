# ---------- Imports ----------
import json
import os
from time import time

from tensorflow import estimator as tf_estimator
from tensorflow.contrib.distribute import CollectiveAllReduceStrategy
from tensorflow.keras import estimator as k_estimator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------- Shape the CNN ----------
# Initialising the CNN as sequential model
classifier = Sequential()

# Convolution Layer
# Transform input in a feature map
# Conv2D        - Two dimensional input (Image) with 4 Parameter
# 32            - Amount of Filters to use
# (3, 3)        - shape of each Filter
# (64, 64, 3)   - Shape of Input (Rows, Columns, Channels)
# 'relu'        - Activation function
classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(Conv2D(32, (3, 3), activation='relu'))

# Pooling Layer
# Reduce the size of the input data by 75%
# (2, 2) - Map 2x2 inputs to one output
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Dropout Layer
# Remove randomly some nodes to add noise
classifier.add(Dropout(0.25))

# Convolution Layer
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

# Flattening Layer
# Maps a X by X Matrix to one Vector
classifier.add(Flatten())

# Full connection Layer
# units  - Amound of nodes in the hidden layer
classifier.add(Dense(units=512, activation='relu'))
classifier.add(Dropout(0.5))

# Output Layer
# units  - Amount of output classes
classifier.add(Dense(units=10, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------- Load Data ----------

# Generator to create batches of data from inputs
# rescale           - Rescaling factors, it means that the RGB input with range 0-255 will be mapped to 0-1 values
# rotation_range    - randomly rotate input by a degree
# shear_range       - randomly apply shearing transformation
# zoom_range        - randomly zooming into the input
# horizontal_flip   - randomly flip the input horizontally
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=0,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
# Generate test data
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load the training data from directory, where each subdirectory is one class
# target_size   - resize images to the given size
# batch_size    - Amount of Images to load for one Step in training (should fit into system memory)
# class_mode    - Set classes to 2D one-hot encoded labels
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='categorical')

# Load the test data from directory, where each subdirectory is one class
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')

# ---------- Start Training ----------
# Configure Tensorboard for Keras with defined log name
tensorboard = TensorBoard(log_dir="logs/{}".format(time()),
                          histogram_freq=0,
                          write_graph=True, write_images=True)


# Train the cnn with given training_set
# steps_per_epoch   -  50.000 Images in total which are split in Batches of 32 Images makes 1562,5 Steps
# epochs            - One epoch equal training one time on the whole dataset (or all steps)
# validation_steps  - 10.000 Images total which are split in Batches of 32 Images makes 312 Steps
# callback          - gives information of training to the tensorboard
# classifier.fit_generator(training_set,
#                          steps_per_epoch=1562,
#                          epochs=25,
#                          validation_data=test_set,
#                          validation_steps=312,
#                          callbacks=[tensorboard])

def training_input_fn():
    return training_set


def test_input_fn():
    return test_set


def model_main():
    distribution = CollectiveAllReduceStrategy(num_gpus_per_worker=1)
    run_config = tf_estimator.RunConfig(train_distribute=distribution, eval_distribute=distribution)
    keras_estimator = k_estimator.model_to_estimator(
        keras_model=classifier, config=run_config, model_dir='./model')
    train_spec = tf_estimator.TrainSpec(input_fn=training_input_fn)
    eval_spec = tf_estimator.EvalSpec(input_fn=test_input_fn)
    tf_estimator.train_and_evaluate(keras_estimator, train_spec, eval_spec)


os.environ["TF_CONFIG"] = json.dumps({

    "cluster": {
        "chief": ["127.0.0.1:2222"],
        "worker": []
    },
    "task": {"type": "chief", "index": 0}
})

# Call the model_main function defined above.
model_main()
