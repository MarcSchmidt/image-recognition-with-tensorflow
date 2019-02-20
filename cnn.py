# ---------- Imports ----------
import os

import tensorflow as tf

import kubernetes_resolver
import load_images

IMAGE_INPUT = None


def create_model(input_shape=(32, 32, 3), start_filters=32,
                 kernel_size=(3, 3), activation='relu',
                 pool_size=(2, 2), output_classes=10):
  # ---------- Shape the CNN ----------
  # Initialising the CNN as sequential model
  model = tf.keras.models.Sequential()

  # Convolution Layer
  # Transform input in a feature map
  # Conv2D        - Two dimensional input (Image) with 4 Parameter
  # 32            - Amount of Filters to use
  # (3, 3)        - shape of each Filter
  # (64, 64, 3)   - Shape of Input (Rows, Columns, Channels)
  # 'relu'        - Activation function
  model.add(tf.keras.layers.Conv2D(start_filters, kernel_size, padding='same',
                                   activation=activation, input_shape=input_shape))
  model.add(tf.keras.layers.Conv2D(start_filters, kernel_size, padding='same',
                                   activation=activation))

  # Pooling Layer
  # Reduce the size of the input data by 75%
  # (2, 2) - Map 2x2 inputs to one output
  model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))
  # Dropout Layer
  # Remove randomly some nodes to add noise
  model.add(tf.keras.layers.Dropout(0.25))

  # Convolution Layer
  model.add(tf.keras.layers.Conv2D(start_filters * 2, kernel_size, padding='same',
                                   activation=activation))
  model.add(tf.keras.layers.Conv2D(start_filters * 2, kernel_size, padding='same',
                                   activation=activation))
  model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))
  model.add(tf.keras.layers.Dropout(0.25))

  # Flattening Layer
  # Maps a X by X Matrix to one Vector
  model.add(tf.keras.layers.Flatten())

  # Full connection Layer
  # units  - Amound of nodes in the hidden layer
  model.add(tf.keras.layers.Dense(units=512, activation=activation))
  model.add(tf.keras.layers.Dropout(0.5))

  # Output Layer
  # units  - Amount of output classes
  model.add(tf.keras.layers.Dense(units=output_classes, activation='softmax'))

  # Compiling the CNN
  optimizer = tf.train.AdamOptimizer()
  model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                metrics=['accuracy'])

  model.summary()

  return model


# ---------- Input function ----------
def input_fn(img=None,
    label=None,
    batch_size=32,
    num_epochs=None,
    num_workers=3,
    worker_index=None,
    shuffle=True):
  data_set = tf.data.Dataset.from_tensor_slices((img, label))

  if worker_index:
    data_set = data_set.shard(num_workers, worker_index)

  if shuffle:
    data_set = data_set.shuffle(buffer_size=batch_size * num_workers)

  data_set = data_set.repeat(num_epochs)
  data_set = data_set.batch(batch_size=batch_size)
  return data_set


def model_main():
  print("--------------------- Load Kubernetes Config ---------------------")
  tf_config = kubernetes_resolver.build_config()
  os.environ['TF_CONFIG'] = str(tf_config)
  worker_index = kubernetes_resolver.fetch_task_index()
  num_workers = len(kubernetes_resolver.build_worker_list())

  #worker_index = None
  #num_workers = 3

  print("--------------------- Load Data ---------------------")
  (x_train, y_train), (x_test, y_test) = load_images.load()

  print("--------------------- Set RunConfiguration ---------------------")
  distribution = tf.contrib.distribute.CollectiveAllReduceStrategy(
      num_gpus_per_worker=1)
  run_config = tf.estimator.RunConfig(train_distribute=distribution,
                                      eval_distribute=distribution)

  # Create estimator
  print("--------------------- Create Estimator ---------------------")
  keras_estimator = tf.keras.estimator.model_to_estimator(
      keras_model=create_model(), config=run_config, model_dir='./model')

  train_spec = tf.estimator.TrainSpec(
      input_fn=lambda: input_fn(img=x_train, label=y_train,
                                num_workers=num_workers,
                                worker_index=worker_index,
                                shuffle=True),
      max_steps=1000)
  eval_spec = tf.estimator.EvalSpec(
      input_fn=lambda: input_fn(img=x_test, label=y_test,
                                num_workers=num_workers,
                                worker_index=worker_index,
                                shuffle=False),
      steps=100)

  # Create estimator
  print("--------------------- Start Training ---------------------")
  tf.estimator.train_and_evaluate(keras_estimator, train_spec, eval_spec)
  print("--------------------- Finish training ---------------------")

  print("--------------------- Start Export ---------------------")
  export_dir = keras_estimator.export_savedmodel(export_dir_base="./dist",
                                                 serving_input_receiver_fn=serving_input_fn)

  print("--------------------- Finish Export on Path ---------------------")

  print("--------------------- Start Tensorboard ---------------------")
  if "TF_CONFIG" in os.environ:
    config = os.environ['TF_CONFIG']
    if "\"type\": \"chief\"" in config:
      os.system('tensorboard --logdir=/notebooks/app/model --port=6006')


def serving_input_fn():
  features = {
    'conv2d_input': tf.placeholder(tf.float32, [None, 32, 32, 3])
  }
  return tf.estimator.export.ServingInputReceiver(features,
                                                  features)


# Define the evironment variable, for local usage
# os.environ[
#   "TF_CONFIG"] = '{"cluster": {"chief": ["localhost:2223"],"worker": ["localhost:2222"]},"task": {"type": "chief", "index": 0}}'

# Call the model_main function defined above.
print("Run Tensorflow")
tf.logging.set_verbosity(tf.logging.DEBUG)
model_main()
