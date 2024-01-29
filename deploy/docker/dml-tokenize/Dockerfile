# Use an official PyTorch runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

RUN apt update && apt install -y build-essential libssl-dev libffi-dev python3-dev

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1

# Run train.py when the container launches
CMD ["python", "./dml_tokenize.py"]
