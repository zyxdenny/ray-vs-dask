FROM rayproject/ray

COPY main.py download-vit.py ./
RUN ["conda", "install", "-y", "pytorch"]
RUN ["conda", "install", "-y", "torchvision"]
RUN ["conda", "install", "-y", "transformers"]
RUN ["conda", "install", "-y", "pillow"]
RUN ["python", "download-vit.py"]
