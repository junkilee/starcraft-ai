# 10.1-cudnn7-runtime-ubuntu18.04 https://gitlab.com/nvidia/cuda/blob/ubuntu18.04/10.0/runtime/cudnn7/Dockerfile
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/dockerfiles/dockerfiles/gpu.Dockerfile
# https://github.com/ContinuumIO/docker-images/blob/master/miniconda3/Dockerfile
ARG UBUNTU_VERSION=16.04
#ARG UBUNTU_VERSION=18.04

FROM ubuntu:${UBUNTU_VERSION} as base

SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        wget

ENV CONDA_VERSION 4.5.12
ENV CONDA_MD5 866ae9dff53ad0874e1d1a60b1ad1ef8

# Create non-root user, install dependencies, install Conda
RUN useradd --create-home --shell /bin/bash --user-group anaconda && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-$CONDA_VERSION-Linux-x86_64.sh && \
    echo "${CONDA_MD5}  Miniconda3-$CONDA_VERSION-Linux-x86_64.sh" > miniconda.md5 && \
    if [ $(md5sum -c miniconda.md5 | awk '{print $2}') != "OK" ] ; then exit 1; fi && \
    mv Miniconda3-$CONDA_VERSION-Linux-x86_64.sh miniconda.sh && \
    mkdir -p /opt && \
    sh ./miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh miniconda.md5 && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    chown -R anaconda:anaconda /opt && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/anaconda/.bashrc && \
    echo "conda activate base" >> /home/anaconda/.bashrc

USER anaconda
ENV PATH "/opt/conda/bin:${PATH}"
ARG TF_PACKAGE=tensorflow
ARG TF_PACKAGE_VERSION=1.12.0
RUN conda install -y python=3.6 && \
    pip install ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}
CMD echo $PATH
