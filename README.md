# Steps to run OpenSeeD inference using docker
## Requirements
- Docker
- GPU with drivers installed

## 1. Extract and keep all the scripts in the same folder as the Dockerfile

## 2. Build the docker Image
```
docker build -t openseed_inf .
```

## 3. Run Inference for Image Segmentation
```
docker run --gpus all --rm -v <path-to-you-folder-with-images>:/workspace/OpenSeeD/input -v <path-to-your-destination-folder>:/workspace/OpenSeeD/output openseed_inf image
```

## 3. Run Inference for Video Segmentation
```
docker run --gpus all --rm -v <path-to-you-folder-with-videos>:/workspace/OpenSeeD/input -v <path-to-your-destination-folder>:/workspace/OpenSeeD/output openseed_inf video
```

# Issues encountered:
## 1. Error response from daemon: could not select device driver "" with capabilities: [[gpu]] 
**Resolution**: Docker cannot access the GPU due to NVIDIA Container toolkit not being installed. 

### Steps to install NVIDIA container toolkit: 

- Set distribution as the version of Ubuntu being used. IF you are on Ubuntu 24.04, NVIDIA doesn’t officially support it, hence set it to 22.04.
```bash
distribution=22.04
```
- Add the NVIDIA package repository
```bash
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \ 

  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 
  

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \ 

  sed 's#deb #deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] #' | \ 

  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null 
```

- Update and install the toolkit
```bash
sudo apt update 
sudo apt install –y nvidia-container-toolkit 
```

- Configure Docker to use the NVIDIA runtime 
```bash
sudo nvidia-ctk runtime configure –runtime=docker 
```

- Restart Docker 
```bash
sudo systemctl restart docker
```

## 2. Error response from daemon: failed to create task for container: failed to create shim task: OCI runtime create failed: runc create failed: unable to start container process: error during container init: error running prestart hook #0: exit status 1, stdout: , stderr: Auto-detected mode as 'legacy'. nvidia-container-cli: initialization error: load library failed: libnvidia-ml.so.1: cannot open shared object file: no such file or directory: unknown 
**Resolution**: Host System is missing the NVIDIA driver or runtime libraries 

- Install the NVIDIA driver: 
```bash
sudo apt install nvidia-driver-535
sudo reboot 
```

- Confirm if its installed with `nvidia-smi` command 
