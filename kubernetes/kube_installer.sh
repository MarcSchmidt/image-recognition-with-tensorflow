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

kubeadm join 10.0.0.6:6443 --token cqxg24.naak7osfr82mz3l1 --discovery-token-ca-cert-hash sha256:045beb44861c57afb40dc9f522ad7a345e657bdfeb172389b2918e61182ebbbc

