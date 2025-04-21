FROM ubuntu:22.04
ADD jax /opt/jax
RUN apt-get update && apt-get install -y python-is-python3 python3-pip
RUN pip install -e /opt/jax[k8s]