FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
MAINTAINER jsk <jsk1107@cj.net>

# 한국어 Locale
RUN apt-get update && apt-get install -y locales
RUN locale-gen ko_KR.UTF-8
ENV LC_ALL ko_KR.UTF-8

# install ubuntu package
RUN apt-get install -y git vim

# Download source code from GIT
RUN mkdir -p /workspace

RUN cd /workspace && \
    git clone "https://github.com/tmdrb0707/pytorch_detection.git"

WORKDIR /workspace/pytorch_detection/

CMD "/bin/bash"
