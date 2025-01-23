FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

# Avoid interactive dialog during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies including those needed for pyenv
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

# Install pyenv
RUN curl https://pyenv.run | bash

# Add pyenv to PATH
ENV HOME /root
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

# Install Python 3.8.18
RUN pyenv install 3.8.18 && \
    pyenv global 3.8.18 && \
    eval "$(pyenv init -)"

# Verify Python version
RUN python --version

# Update pip and setuptools
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install Python dependencies with compatible versions
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    pandas==1.5.3 \
    biopython==1.81 \
    sentencepiece==0.1.99 \
    transformers==4.30.1 \
    modelcif==0.7

# Install PyG dependencies separately with specific versions
RUN pip install torch-scatter==2.1.0+pt112cu113 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html && \
    pip install torch-sparse==0.6.15+pt112cu113 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html && \
    pip install torch-cluster==1.6.0+pt112cu113 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html && \
    pip install torch-geometric==2.3.0

# Install ESMFold and dllogger
RUN pip install --no-cache-dir "fair-esm[esmfold]" && \
    pip install --no-cache-dir 'dllogger @ git+https://github.com/NVIDIA/dllogger.git'

# Copy GeoPoc repository including openfold
COPY GeoPoc /app/GeoPoc

# Install OpenFold from local copy
WORKDIR /app/GeoPoc/openfold/openfold-main
RUN python setup.py install

# Set executable permission for mkdssp
WORKDIR /app/GeoPoc
RUN chmod +x /app/GeoPoc/script/feature_extraction/mkdssp

# Add environment variable for CUDA device (can be overridden at runtime)
ENV CUDA_VISIBLE_DEVICES=0

# Command to run predictions (can be overridden at runtime)
ENTRYPOINT ["python", "predict.py"]
CMD ["-h"]