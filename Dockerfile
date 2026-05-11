# This Dockerfile aims to build the base image for Deepfakbench.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

LABEL maintainer="Deepfake"

# Install dependencies outside of the base image
RUN DEBIAN_FRONTEND=noninteractive apt update && \
	apt install -y --no-install-recommends automake \
    build-essential  \
    ca-certificates  \
    libfreetype6-dev  \
    libtool  \
    pkg-config  \
    python3-dev  \
    python3-pip \
    cmake \
	&& \
    rm -rf /var/lib/apt/lists/*

WORKDIR /

# Install Python dependencies
RUN pip install --no-cache-dir certifi setuptools \
    && \
    pip --no-cache-dir install dlib==19.24.6\
    imageio==2.36.1\
    imgaug==0.4.0\
    scipy==1.14.1\
    seaborn==0.13.2\
    pyyaml==6.0.2\
    imutils==0.5.4\
    opencv-python==4.11.0.86\
    scikit-image==0.24.0\
    scikit-learn==1.6.1\
    efficientnet-pytorch==0.7.1\
    timm==1.0.12\
    segmentation-models-pytorch==0.3.4\
    torchtoolbox==0.1.8.2\
    tensorboard==2.18.0\
    setuptools==75.8.0\
    loralib\
    pytorchvideo\
    einops\
    transformers\
    filterpy\
    simplejson\
    kornia\
    git+https://github.com/openai/CLIP.git

ENV MODEL_NAME=deepfakebench

# Expose port
EXPOSE 6000
