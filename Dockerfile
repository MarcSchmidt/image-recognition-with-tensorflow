FROM tensorflow/tensorflow:1.12.0-rc1-py3
RUN mkdir app
WORKDIR app
COPY cnn.py .
COPY dataset dataset
RUN mkdir model


ENTRYPOINT ["python", "cnn.py"]