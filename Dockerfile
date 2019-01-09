FROM tensorflow/tensorflow:1.12.0-py3

RUN mkdir app
WORKDIR app
RUN mkdir model
RUN pip install kubernetes
COPY cnn.py .
COPY load_images.py .
COPY kubernetes_resolver.py .
COPY dataset/test_set dataset/test_set
COPY dataset/training_set dataset/training_set

ENTRYPOINT ["python", "cnn.py"]
