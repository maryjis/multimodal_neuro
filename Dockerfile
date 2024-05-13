# adapted from https://github.com/sinzlab/pytorch-docker/blob/master/Dockerfile
# and https://fabiorosado.dev/blog/install-conda-in-docker/
FROM nvidia/cuda:11.4.0-runtime-ubuntu18.04 as base

# Install base utilities
RUN apt-get update && \
    apt-get install -y build-essential wget vim git && \
    apt-get clean && \
    # best practice to keep the Docker image lean
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install miniconda (yes) with required python version
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# install graph tools
COPY ./graph-tool_install.sh /tmp
RUN /tmp/graph-tool_install.sh


COPY ./install_pyg.sh /tmp
COPY ./requirements.txt /tmp
COPY . app

# install requirements; force reinstall scipy to get rid of bug w/ glibc
# (change cuda to cpu if you want to)
RUN /tmp/install_pyg.sh cu113 && python -m pip install -r /tmp/requirements.txt && pip install -I scipy==1.10.1
# install neurograph
RUN pip install -e /app
# install the fork of cwn
RUN mkdir /cwn && git clone https://github.com/gennadylaptev/cwn.git cwn && pip install -e cwn

# Deal with pesky Python 3 encoding issue
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive
ENV MPLLOCALFREETYPE 1

WORKDIR /app

# 'fix' issues w/ git
RUN git config --global --add safe.directory /app

# Export port for Jupyter Notebook
EXPOSE 8888
