# Use the official PyTorch image as base
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Set non-interactive mode and timezone
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

# Configure system and install dependencies in a single RUN command
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    sed -i 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
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
        lld lldb clangd openssh-server && \
    ccache -M 10G && \
    mkdir -p /var/run/sshd && \
    sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config && \
    echo "root:passwd" | chpasswd && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Clone the Triton repository with shallow clone
RUN git clone --depth 1 --branch beta2 https://github.com/TruthK/triton-shared.git && \
    cd triton-shared && \
    git config --global http.postBuffer 524288000 && \
    git submodule update --init --depth=1

WORKDIR /workspace/triton-shared


# Expose SSH port
EXPOSE 20001

# Start SSH service when container launches
CMD ["/usr/sbin/sshd", "-D"]


# Optical
# Create a development conda environment with packages such as torch2.6.0 pybind11 (environment name :triton_shared_mlir_nv). If there is no torch in the base environment, the script default pytorch-cuda version is 12.1. If other versions are required, modify the input parameters
# RUN chmod +x create_conda_env.sh
# RUN bash create_conda_env.sh cu124
# RUN conda init bash && \
#     echo "conda activate triton_shared_mlir_nv" >> ~/.bashrc
# RUN /bin/bash -c "source ~/.bashrc"
