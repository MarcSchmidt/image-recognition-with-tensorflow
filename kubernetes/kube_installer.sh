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

kubeadm join 10.0.0.20:6443-token az7353.7o62zkcy7lq577ph -discovery-token-cb4f62ce62e5ab4fe1a472372993ed49eecce453b938a7d41902d1ac6a6ecc457

