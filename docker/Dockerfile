#FROM gcr.io/tensorflow/tensorflow
FROM gcr.io/tensorflow/tensorflow:latest-gpu
#FROM debian:jessie

ADD requirements.txt /root

#RUN apt-get update && \
#    apt-get install -y python-pip python-dev && \
#    apt-get -y autoremove && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN pip install -r /root/requirements.txt ##&& \
    rm -rf /root/.cache/pip/*

#RUN apt-get update && apt-get install -y --no-install-recommends \
#        build-essential \
#        curl \
#        libfreetype6-dev \
#        libpng12-dev \
#        libzmq3-dev \
#        pkg-config \
#        rsync \
#        software-properties-common \
#        unzip \
#        && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*
#
#RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
#    python get-pip.py && \
#    rm get-pip.py
#
#RUN pip --no-cache-dir install \
#    http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl

#RUN apt-get -y remove python-pip && easy_install pip && \
#    ln /usr/local/bin/pip /usr/bin/pip && \
#    pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp27-none-linux_x86_64.whl
#    pip install tensorflow-gpu

#RUN pip install tensorflow

ENV PATH="/usr/local/cuda/bin:/usr/local/nvidia/bin:/usr/lib/x86_64-linux-gnu:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/nvidia:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

WORKDIR /gan-lib/