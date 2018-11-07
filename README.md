# image-recognition-with-tensorflow
A project to train a convolutional neural network with keras on a MacOS Cluster with the help Docker & Kubernetes.

# Train the Model
To train the model just execute the python script.
```bash
python cnn.py
```
The performance of the Model can be viewed with Tensorboard.
```bash
tensorboard --logdir=logs/
```

# Convert images to tfrecords
```bash
python create_tfrecords/create_tfrecord.py --dataset_dir=dataset_dir --tfrecord_filename=tfrecord_filename
```
