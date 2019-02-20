#!/usr/bin/env bash
docker build -t docker.nexus.archi-lab.io/archilab/tensorflow-img-rec ../
docker push docker.nexus.archi-lab.io/archilab/tensorflow-img-rec
kubectl -n tensorflow delete po,svc,statefulsets,deployments  --all
kubectl apply -f ./deployment.yaml
