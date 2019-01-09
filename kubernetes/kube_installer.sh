#!/usr/bin/env bash
apt-get update && apt-get install -y apt-transport-https curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
cat <<EOF >/etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF
apt-get update
apt-get install -y kubelet kubeadm kubectl docker.io
apt-mark hold kubelet kubeadm kubectl
systemctl enable docker.service
swapoff -a
kubeadm config images pull

kubeadm join 10.0.0.20:6443 --token 3x16gl.td7jti3xnrd27uwi --discovery-token-ca-cert-hash sha256:cc6689570cf0a9ae4c84a2ae0c557e29fc44ad89f092b27e89ab22a7f10b5f82

