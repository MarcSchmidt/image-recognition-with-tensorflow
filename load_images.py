import os
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def load():

  train_dir = Path('dataset/training_set')
  test_dir = Path('dataset/test_set')

  # for later use to map back label ids to label names
  dict = {}

  x_train = []
  y_train = []

  for i, dir in enumerate(train_dir.iterdir()):
    if dir.is_dir():
      label = os.path.basename(str(dir))
      dict[i] = label
      for img in dir.iterdir():
        x_train.append(img_to_array(load_img(img)))
        y_train.append(i)

  x_train = np.array(x_train) / 255.0
  y_train = np.array(y_train)
  y_train = y_train.reshape(y_train.shape + (1,))

  x_test = []
  y_test = []

  for i, dir in enumerate(test_dir.iterdir()):
    if dir.is_dir():
      for img in dir.iterdir():
        x_test.append(img_to_array(load_img(img)))
        y_test.append(i)

  x_test = np.array(x_test) / 255.0
  y_test = np.array(y_test)
  y_test = y_test.reshape(y_test.shape + (1,))

  return (x_train, y_train), (x_test, y_test)
