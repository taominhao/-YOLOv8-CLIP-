import os
import requests
from tqdm import tqdm
from bs4 import BeautifulSoup

TM_DIED_URL = 'https://www.flickr.com/gp/73847677@N02/GRn3G6'
SAVE_DIR = 'datasets/tmdied/'

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
}

def get_image_urls():
    resp = requests.get(TM_DIED_URL, headers=headers)
    soup = BeautifulSoup(resp.text, 'html.parser')
    imgs = soup.find_all('img')
    urls = []
    for img in imgs:
        src = img.get('src')
        if src and 'staticflickr.com' in src:
            # 自动补全https:前缀
            if src.startswith('//'):
                src = 'https:' + src
            urls.append(src)
    # 去重，全部下载
    urls = list(dict.fromkeys(urls))
    return urls

def download_images(urls):
    os.makedirs(SAVE_DIR, exist_ok=True)
    for i, url in enumerate(tqdm(urls, desc='下载图片')):
        ext = url.split('.')[-1].split('?')[0]
        save_path = os.path.join(SAVE_DIR, f'{i+1:03d}.{ext}')
        if os.path.exists(save_path):
            continue
        try:
            img_data = requests.get(url, headers=headers).content
            with open(save_path, 'wb') as f:
                f.write(img_data)
        except Exception as e:
            print(f'下载失败: {url}, 错误: {e}')

if __name__ == '__main__':
    print('正在获取图片链接...')
    urls = get_image_urls()
    print(f'共获取到{len(urls)}张图片，开始下载...')
    download_images(urls)
    print('下载完成！') 