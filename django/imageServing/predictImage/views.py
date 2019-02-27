import time

import numpy as np
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from grpc.beta import implementations
from scipy.misc import imread
from tensorflow.contrib.util import make_tensor_proto
# needs pip install tensorflow-serving-api
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def index(request):
  filename = "TestImage.png"
  if request.method == 'POST' and request.FILES['myfile']:
    myfile = request.FILES['myfile']
    fs = FileSystemStorage()
    fs.delete(filename)
    fs.save(filename, myfile)
    uploaded_file_url = fs.url(filename)
    return render(request, 'predictImage/index.html', {
      'uploaded_file_url': uploaded_file_url,
    })

  if request.method == 'GET' and request.GET.get('predictImage'):
    data = run(host='localhost', port=8500, image='./media/' + filename,
               model='model', signature_name='serving_default')
    fs = FileSystemStorage()
    uploaded_file_url = fs.url(filename)
    return render(request, 'predictImage/index.html', {
      'uploaded_file_url': uploaded_file_url,
      'data': data
    })

  return render(request, 'predictImage/index.html')


def run(host, port, image, model, signature_name):
  # channel = grpc.insecure_channel('%s:%d' % (host, port))
  channel = implementations.insecure_channel(host, port)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  # Read an image
  data = imread(image)
  data = data.astype(np.float32)
  data = data / 255.0
  # print(data)

  start = time.time()

  # Call classification model to make prediction on the image
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model
  request.model_spec.signature_name = signature_name
  request.inputs['conv2d_input'].CopyFrom(
      make_tensor_proto(data, shape=[1, 32, 32, 3]))

  data = stub.Predict(request, 10.0)

  end = time.time()
  time_diff = end - start

  print('Raw data: ')
  print(data)

  result = []
  classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
             'horse', 'ship', 'truck']
  for index in range(0, len(classes)):
    result.append(
        [classes[index], data.outputs['dense_1'].float_val[index] * 100])

  print('time elapased: {}'.format(time_diff))
  return result
