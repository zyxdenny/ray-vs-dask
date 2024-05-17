import ray
from typing import Any, Dict
import numpy as np
from transformers import AutoImageProcessor, ViTMAEForPreTraining
from PIL import Image
import torch

root = "s3://imagenet-c-m/images/"
ds = ray.data.read_images(root)

def transform(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-base')
    batch["image"] = processor(images=batch["image"], return_tensors="pt")
    return batch

class ImageClassifier:
    def __init__(self):
        self.model = ViTMAEForPreTraining.from_pretrained('facebook/vit-mae-base')

    def __call__(self, batch):
        inputs = torch.from_numpy(batch["image"])
        outputs = self.model(**inputs)
        return outputs

predictions = ds.map_batches(
    ImageClassifier,
    concurrency=2,
    batch_size=4 # Change batch size here
)