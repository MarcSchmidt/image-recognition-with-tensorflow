import json
import os
import re

from kubernetes import client, config

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
    task = {'type': os.environ.get("POD_TASK"), 'index': fetch_task_index()}
    cluster = {'chief': build_chief_list(), 'worker': build_worker_list()}
    tf_config = {'cluster': cluster, 'task': task}
    print(tf_config)
    return json.dumps(tf_config)


def build_worker_list(namespace="tensorflow"):
    worker_nodes = []
    pods = k8.list_namespaced_pod(namespace, watch=False)
    for item in pods.items:
        if re.match("tensorflow-worker-([0-9]+)", item.metadata.name):
            node_name = "%s.worker-svc.tensorflow.svc.cluster.local" \
                        % item.metadata.name
            node_port = item.spec.containers[0].ports[0].container_port
            worker_nodes.append("%s:%s" % (node_name, node_port))
    return worker_nodes


def fetch_task_index():
    if os.environ.get("POD_TASK") == "worker":
        pod_name = os.environ.get("POD_NAME")
        return int(pod_name.split("-")[2])
    return 0


def build_chief_list(namespace="tensorflow"):
    chief_nodes = []
    services = k8.list_namespaced_service(namespace, watch=False)
    for item in services.items:
        if "chief" in item.metadata.name:
            chief_nodes.append(
                    "%s:%s" % (item.metadata.name, item.spec.ports[0].port))
    return chief_nodes
