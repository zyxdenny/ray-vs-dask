# include PyTorch  CUDA
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# workdir
WORKDIR /app

# env
COPY requirements.txt /app/
RUN pip install -r requirements.txt

#
COPY . /app/