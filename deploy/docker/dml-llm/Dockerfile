FROM nvidia/cuda:12.3.1-base-ubuntu22.04
# FROM nvidia/cuda:12.3.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# # Install system dependencies
# RUN apt-get update && \
#     apt-get install -y \
#         git \
#         python3-pip \
#         python3-dev \
#         libglib2.0-0
RUN apt-get update && \
    apt-get install -y \
        python3-pip \
        python3-dev
#         libglib2.0-0

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# RUN python3 -m pip install --upgrade pip

#RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html
RUN pip install torch --index-url https://download.pytorch.org/whl/cu121

ENV PYTHONUNBUFFERED=1

# Run train.py when the container launches
CMD ["python", "./train.py"]
