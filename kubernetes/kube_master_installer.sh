#!/usr/bin/env bash

# Update an install util packages. These are needed for further installations
apt-get update && apt-get install -y apt-transport-https curl

# Download a apt-key from a special repository inorder to install kuberenetes
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
cat <<EOF >/etc/apt/sources.list.d/kubernetes.list
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF

# Update repositories and install kubernetes and docker
apt-get update
apt-get install -y kubelet kubeadm kubectl docker.io

# Marks kubernetes as a manual installation
apt-mark hold kubelet kubeadm kubectl

# Enable docker as a service
systemctl enable docker.service

# Turn swap memory off due to kubernetes requirements
swapoff -a

# Configure kubernetes
kubeadm config images pull

# Initialize the Master-Node, insert the master-node IP-address
kubeadm init --apiserver-advertise-address=10.0.0.20 --apiserver-cert-extra-sans=10.0.0.20

# Create further configurations
mkdir -p $HOME/.kube
cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
chown $(id -u):$(id -g) $HOME/.kube/config

#Install networkdriver for communication between different nodes
kubectl -n kube-system apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml