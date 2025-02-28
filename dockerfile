FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# Set non-interactive mode to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set the timezone
RUN export TZ=Etc/UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime
RUN echo $TZ > /etc/timezone

# Update the system package list and upgrade existing packages
RUN sed -i 's/http:\/\/archive.ubuntu.com/http:\/\/mirrors.aliyun.com/g' /etc/apt/sources.list
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get autoremove

# Update the system package list and install basic tools
RUN apt-get install -y --fix-missing \
    build-essential \
    curl \
    g++ gcc \
    git \
    graphviz \
    libopenblas-dev \
    ninja-build \
    pkg-config \
    unzip \
    wget \
    vim \
    clang-format \
    sudo \
    gdb \
    ccache \
    cmake \
    lsb-release \
    software-properties-common \
    gnupg \
    zlib1g-dev \
    libzstd-dev \
    clang \
    lld openssh-server

RUN ccache -M 30G

RUN mkdir /var/run/sshd
RUN sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

# Expose SSH port
EXPOSE 20001

# Set the root password
RUN echo "root:passwd" | chpasswd

# Clone the Triton repository
RUN git clone https://github.com/TruthK/triton-shared.git && \
    cd triton-shared && \
    git fetch && \
    git checkout kzx_nv

WORKDIR /workspace/triton-shared

RUN git config --global http.postBuffer 524288000
RUN git submodule update --init

# CCreate a development conda environment with packages such as torch2.6.0 pybind11 (environment name :triton_shared_mlir_nv). If there is no torch in the base environment, the script default pytorch-cuda version is 12.1. If other versions are required, modify the input parameters

# Start SSH service when container launches
CMD ["/usr/sbin/sshd", "-D"]