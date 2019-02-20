# !bin/bash

nodes=$(kubectl get nodes -o jsonpath='{range.items[*].metadata}{.name} {end}')
for n in $nodes; do sudo scp ~/.docker/config.json kube@$n:/var/lib/kubelet/config.json; done