import os
import requests
from tqdm import tqdm
import zipfile
import open_clip
import torch

DATASET_URL = 'https://data.vision.ee.ethz.ch/cvl/mahmoudnafifi/Exposure_Correction/dataset.zip'
SAVE_PATH = 'datasets/exposure/dataset.zip'
EXTRACT_PATH = 'datasets/exposure/'

def download_file(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(save_path, 'wb') as file, tqdm(
        desc=save_path,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

model_name = 'ViT-B-32'
cache_path = '/Users/taomh/Desktop/bysj/open_clip/ViT-B-32-openai'

model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained='',  # 不自动下载
)
state_dict = torch.load(cache_path, map_location='cpu')
model.load_state_dict(state_dict)

if __name__ == '__main__':
    if not os.path.exists(SAVE_PATH):
        print('正在下载Flickr Exposure Dataset...')
        download_file(DATASET_URL, SAVE_PATH)
    else:
        print('数据集已存在，跳过下载。')
    print('正在解压数据集...')
    unzip_file(SAVE_PATH, EXTRACT_PATH)
    print('数据集准备完毕！')