import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_DIR = 'datasets/tmdied/'
HIST_DIR = os.path.join(IMG_DIR, 'hist')
RESULTS_FILE = 'exposure_results.txt'

# 曝光判断阈值（可根据实际情况调整）
UNDER_EXPOSED_THRESHOLD = 60   # 欠曝：平均亮度低于此值
OVER_EXPOSED_THRESHOLD = 190  # 过曝：平均亮度高于此值


def analyze_exposure(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return '无法读取', None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    # 判断曝光类型
    if mean_brightness < UNDER_EXPOSED_THRESHOLD:
        status = '欠曝'
    elif mean_brightness > OVER_EXPOSED_THRESHOLD:
        status = '过曝'
    else:
        status = '正常'
    return status, mean_brightness, gray


def save_histogram(gray, img_file, status):
    plt.figure(figsize=(6, 4))
    plt.hist(gray.ravel(), bins=256, range=(0, 256), color='gray', alpha=0.7)
    plt.title(f'{img_file} - {status}')
    plt.xlabel('像素亮度值')
    plt.ylabel('像素数量')
    plt.tight_layout()
    os.makedirs(HIST_DIR, exist_ok=True)
    save_path = os.path.join(HIST_DIR, f'hist_{os.path.splitext(img_file)[0]}.jpg')
    plt.savefig(save_path)
    plt.close()


def main():
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    results = []
    for img_file in img_files:
        img_path = os.path.join(IMG_DIR, img_file)
        status, brightness, gray = analyze_exposure(img_path)
        results.append((img_file, status, brightness))
        print(f'{img_file}: {status} (平均亮度: {brightness:.2f} if brightness else "N/A")')
        if gray is not None:
            save_histogram(gray, img_file, status)
    # 保存结果
    with open(RESULTS_FILE, 'w') as f:
        for img_file, status, brightness in results:
            f.write(f'{img_file}: {status} (平均亮度: {brightness})\n')
    print(f'分析完成，结果已保存到 {RESULTS_FILE}')
    print(f'所有直方图已保存到 {HIST_DIR}/')

if __name__ == '__main__':
    main() 