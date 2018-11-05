FROM tensorflow/tensorflow:1.12.0-rc1-py3
RUN mkdir app
WORKDIR app
COPY cnn.py .
COPY kubernetesResolver.py .
RUN pip install kubernetes
RUN mkdir /root/.keras
RUN mkdir /root/.keras/datasets
COPY dataset/cifar-10-batches-py.tar.gz /root/.keras/datasets
RUN mkdir model


ENTRYPOINT ["python", "cnn.py"]