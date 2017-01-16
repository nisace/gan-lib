FROM gcr.io/tensorflow/tensorflow

ADD requirements.txt /root
RUN pip install -r /root/requirements.txt && \
    rm -rf /root/.cache/pip/*

WORKDIR /gan-lib/