from kubernetes import client, config
import re

# only works in the cluster itself not from outside or remote.
config.load_incluster_config()

k8 = client.CoreV1Api()


def print_pods_services():
    print("All Pods:")
    ret_pods = k8.list_namespaced_pod("tensorflow", watch=False)
    for i in ret_pods.items:
        print("%s\t%s\t%s" %
              (i.status.pod_ip, i.metadata.namespace, i.metadata.name))

    print("--------------------------------")

    print("All Services:")
    ret_svc = k8.list_namespaced_service("tensorflow", watch=False)
    for i in ret_svc.items:
        print("%s\t%s" %
              (i.metadata.namespace, i.metadata.name))
    print("--------------------------------")


def build_config():
    worker_nodes = []
    pods = k8.list_namespaced_pod("tensorflow", watch=False)
    for item in pods.items:
        print("Try to match %s" % item.metadata.name)
        if re.match("tensorflow-worker-([0-9]+)", item.metadata.name):
            node_name = item.metadata.name
            node_port = item.spec.containers[0].ports[0].container_port
            index = item.metadata.name.split("-")[2]
            worker_nodes.append("%s:%s" % (node_name, node_port))
            print("Matched: %s on port %s" % (node_name, node_port))
    return worker_nodes
