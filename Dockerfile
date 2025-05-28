FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git wget python3 python3-pip ffmpeg libgl1 unzip && \
    rm -rf /var/lib/apt/lists/*

# Symlink python and pip
RUN [ -e /usr/bin/python ] || ln -s /usr/bin/python3 /usr/bin/python && \
    [ -e /usr/bin/pip ] || ln -s /usr/bin/pip3 /usr/bin/pip

# Clone OpenSeeD
RUN git clone https://github.com/abhigoku10/openseed_adas.git

# Set working dir to OpenSeeD
WORKDIR /workspace/openseed_adas

# Install dependencies
RUN pip install -r requirements.txt && \
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 && \
    pip install numpy==1.23.5 && \
    pip install 'git+https://github.com/facebookresearch/detectron2.git' && \
    pip install 'git+https://github.com/cocodataset/panopticapi.git' 
    
#Download model weights
RUN mkdir -p weights && \
    wget -q https://github.com/IDEA-Research/OpenSeeD/releases/download/coco_pano_sota_swinl/openseed_swinl_pano_sota.pt \
         -O weights/openseed_swinl_pano_sota.pt

# Add entrypoint script
#COPY run_inference.sh ./run_inference.sh
RUN chmod +x run_inference.sh

ENTRYPOINT ["./run_inference.sh"]
