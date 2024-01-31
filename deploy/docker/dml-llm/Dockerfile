# FROM nvidia/cuda:12.3.1-base-ubuntu22.04
# FROM nvidia/cuda:12.3.1-devel-ubuntu22.04
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set environment variables
# ENV DEBIAN_FRONTEND=noninteractive
# ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# ENV NVIDIA_REQUIRE_CUDA=cuda>=11.0 brand=tesla,driver>=418,driver<419 brand=tesla,driver>=440,driver<441 brand=tesla,driver>=450,driver<451


# # Install system dependencies
# RUN apt-get update && \
#     apt-get install -y \
#         git \
#         python3-pip \
#         python3-dev \
#         libglib2.0-0
# RUN apt-get update && \
#     apt-get install -y \
#         python3-pip \
#         python3-dev
#         libglib2.0-0

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# RUN python3 -m pip install --upgrade pip

#RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
# RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

ENV PYTHONUNBUFFERED=1

# Run train.py when the container launches
CMD ["python", "./train.py"]
