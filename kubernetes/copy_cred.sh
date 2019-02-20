# !bin/bash

nodes=$(kubectl get nodes -o jsonpath='{range .items[*].status.addresses[?(@.type=="ExternalIP")]}{.address} {end}')
for n in $nodes; do sudo scp ~/.docker/config.json kube@$n:/var/lib/kubelet/config.json; done