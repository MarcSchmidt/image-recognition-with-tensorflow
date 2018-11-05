#!/usr/bin/env bash
docker build -t tensorflow-img-rec ../
kubectl -n tensorflow delete po,svc,statefulsets  --all
kubectl apply -f ./deployment.yaml
