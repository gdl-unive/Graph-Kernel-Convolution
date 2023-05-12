ARG TORCH="torch==1.8.1"
ARG TORCH_GEO="torch-geometric==1.7.0 torch_scatter==2.0.8 torch-sparse==0.6.12"
ARG TORCH_GEO_URL="https://data.pyg.org/whl/torch-1.8.1"
ARG CUDA_VER=cu110


FROM continuumio/miniconda3 as build-py
WORKDIR /root
RUN printf "name: py\n\ndependencies:\n  - python=3.8.12\n  - pip\n" > environment.yml
RUN conda env create -f environment.yml && \
    conda install -c conda-forge conda-pack
RUN /opt/conda/envs/py/bin/pip3 install --no-cache-dir autopep8
COPY requirements.txt /tmp/requirements.txt
COPY requirements2.txt /tmp/requirements2.txt
RUN apt-get update && apt-get install -y --no-install-recommends gcc make cmake g++ && apt-get clean
RUN /opt/conda/envs/py/bin/pip3 install --no-cache-dir -r /tmp/requirements.txt


FROM build-py AS build-py-cpu
ARG TORCH
ARG TORCH_GEO
ARG TORCH_GEO_URL
RUN /opt/conda/envs/py/bin/pip3 install --no-cache-dir $TORCH+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN /opt/conda/envs/py/bin/pip3 install --no-cache-dir $TORCH_GEO -f $TORCH_GEO_URL+cpu.html
RUN /opt/conda/envs/py/bin/pip3 install --no-cache-dir -r /tmp/requirements2.txt
RUN conda-pack -n py -o /tmp/env.tar && \
    mkdir /venv && cd /venv && \
    tar xf /tmp/env.tar


FROM build-py AS build-py-cuda
ARG TORCH
ARG TORCH_GEO
ARG TORCH_GEO_URL
ARG CUDA_VER
COPY requirements2.txt /tmp/requirements2.txt
RUN /opt/conda/envs/py/bin/pip3 install --no-cache-dir $TORCH
RUN /opt/conda/envs/py/bin/pip3 install --no-cache-dir $TORCH_GEO -f $TORCH_GEO_URL+$CUDA_VER.html
RUN /opt/conda/envs/py/bin/pip3 install --no-cache-dir -r /tmp/requirements2.txt
RUN conda-pack -n py -o /tmp/env.tar && \
    mkdir /venv && cd /venv && \
    tar xf /tmp/env.tar


FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS cuda-python
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y --no-install-recommends git && apt-get clean
COPY --from=build-py-cuda /venv /venv
ENV PATH="$PATH:/venv/bin"
CMD [ "/bin/bash" ]


FROM ubuntu:22.04 AS python
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install -y --no-install-recommends git && apt-get clean
COPY --from=build-py-cpu /venv /venv
ENV PATH="$PATH:/venv/bin"
CMD [ "/bin/bash" ]
