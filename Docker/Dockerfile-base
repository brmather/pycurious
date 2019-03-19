FROM alpine:latest

# Install things
RUN echo "http://mirror.leaseweb.com/alpine/edge/testing" >> /etc/apk/repositories
RUN apk add --update --no-cache \
    gcc \
    g++ \
    gfortran \
    build-base \
    python3-dev \
    py3-pip \
    py3-numpy \
    py3-scipy \
    py3-zmq \
    cmake \
    curl \
    wget \
    libzmq \
    zlib-dev \
    musl-dev \
    freetype-dev \
    libpng-dev \
    libxml2-dev \
    openblas-dev \
    geos-dev \
    proj4-dev

RUN echo "http://dl-cdn.alpinelinux.org/alpine/edge/main" >> /etc/apk/repositories
RUN apk add --update --no-cache \
    libressl2.7-libcrypto \
    gdal-dev


RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
# RUN python3 -m pip install --no-cache-dir --upgrade numpy && \
RUN python3 -m pip install --no-cache-dir --upgrade pip numpy && \
    python3 -m pip install --no-cache-dir setuptools wheel Cython && \
    python3 -m pip install --no-cache-dir \
    packaging \
    jupyter \
    pyepsg \
    cartopy && \
    python3 -m pip install --no-cache-dir matplotlib


# Install Tini.. this is required because CMD (below) doesn't play nice with notebooks for some reason:
# https://github.com/ipython/ipython/issues/7062, https://github.com/jupyter/notebook/issues/334
RUN curl -L https://github.com/krallin/tini/releases/download/v0.10.0/tini > tini && \
    echo "1361527f39190a7338a0b434bd8c88ff7233ce7b9a4876f3315c22fce7eca1b0 *tini" | sha256sum -c - && \
    mv tini /usr/local/bin/tini && \
    chmod +x /usr/local/bin/tini

# expose notebook port
EXPOSE 8888


# add a notebook profile
RUN mkdir -p -m 700 /root/.jupyter/ && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py

# note we also use xvfb which is required for viz
ENTRYPOINT ["/usr/local/bin/tini", "--"]