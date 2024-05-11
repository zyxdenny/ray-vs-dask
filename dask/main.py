import time
import torch
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
import dask
import dask.array as da
from dask.distributed import Client
from dask import delayed, compute
from PIL import Image
import numpy as np

# 初始化 Dask 客户端
client = Client('10.244.0.6:8786')  # Dask scheduler address and port

def load_and_preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path)
    image = image.convert('RGB')  # 确保是 RGB
    image_tensor = preprocess(image)
    return image_tensor.numpy()

def inference(batch):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()
    model.to('cuda')
    with torch.no_grad():
        inputs = torch.tensor(batch, device='cuda')
        outputs = model(inputs)
        return outputs.cpu().numpy()

def load_images(image_folder):
    from glob import glob
    image_paths = glob(f'{image_folder}/*.jpg')
    images = [load_and_preprocess_image(path) for path in image_paths]
    return np.stack(images)

def run_image_processing(image_folder):
    start_time = time.time()  # start
    images = load_images(image_temp_folder)
    images_da = da.from_array(images, chunks=(100, 3, 224, 224))  # Example chunk size

    # batch
    results = []
    for i in range(0, len(images_da), 1000):
        batch_results = dask.delayed(inference)(images_da[i:i+1000])
        results.append(batch_results)

    # result
    final_results = dask.compute(*results, scheduler='threads')
    end_time = time.time()  # end

    # 性能输出
    total_time = end_time - start_time
    print(f"Image processing completed in {total_normal_time:.2f} seconds.")
    if len(images_da) > 0:
        print(f"Average time per batch: {total_time / len(images_da):.2f} seconds")

    return final_results

if __name__ == '__main__':
    # start offline batch service
    image_processing_results = run_image_processing("/Users/gongyitong/Downloads/Images")
    print("Image processing results obtained.")
