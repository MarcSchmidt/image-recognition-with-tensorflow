# !bin/bash

nodes=$(kubectl get nodes -o jsonpath='{range.items[*].metadata}{.name} {end}')
for n in $nodes; do scp ~/.docker/config.json root@$n:/var/lib/kubelet/config.json; done