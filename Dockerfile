FROM pytorch/pytorch:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -y update && \
    apt-get -y upgrade && \
    apt-get -y install \
        build-essential \
        wget \
        curl \
        git \
        make \
        gcc \
        graphviz \
        sudo

COPY requirements.txt .
COPY . .

RUN conda update conda
RUN conda update conda-build
RUN conda install pip

RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade -r requirements.txt
