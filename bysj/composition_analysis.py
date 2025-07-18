import os
import cv2
import numpy as np
from ultralytics import YOLO

IMG_DIR = 'datasets/tmdied/'
RESULTS_FILE = 'composition_results.txt'
OUTPUT_DIR = os.path.join(IMG_DIR, 'composition_vis')

# 三分法线的比例
THIRD_RATIO = 1/3

# 检测类别（只分析人物/人类）
PERSON_CLASS_NAMES = ['person', '人']


def is_on_third_line(box, img_shape):
    (h, w) = img_shape[:2]
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    thirds_x = [w * THIRD_RATIO, w * (1 - THIRD_RATIO)]
    thirds_y = [h * THIRD_RATIO, h * (1 - THIRD_RATIO)]
    tol_x = w * 0.1
    tol_y = h * 0.1
    on_x = any(abs(cx - tx) < tol_x for tx in thirds_x)
    on_y = any(abs(cy - ty) < tol_y for ty in thirds_y)
    return on_x or on_y


def main():
    model = YOLO('yolov8n.pt')  # 使用官方nano模型，自动下载
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    img_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    results = []
    for img_file in img_files:
        img_path = os.path.join(IMG_DIR, img_file)
        img = cv2.imread(img_path)
        if img is None:
            results.append((img_file, '无法读取'))
            continue
        h, w = img.shape[:2]
        pred = model(img_path)[0]
        found = False
        for box, cls in zip(pred.boxes.xyxy.cpu().numpy(), pred.boxes.cls.cpu().numpy()):
            class_name = model.model.names[int(cls)]
            if class_name in PERSON_CLASS_NAMES:
                x1, y1, x2, y2 = map(int, box)
                on_third = is_on_third_line((x1, y1, x2, y2), img.shape)
                result = '主体在三分线上' if on_third else '主体不在三分线上'
                results.append((img_file, result))
                # 可视化
                color = (0, 255, 0) if on_third else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, result, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                # 画三分线
                for tx in [int(w*THIRD_RATIO), int(w*(1-THIRD_RATIO))]:
                    cv2.line(img, (tx, 0), (tx, h), (200, 200, 0), 1)
                for ty in [int(h*THIRD_RATIO), int(h*(1-THIRD_RATIO))]:
                    cv2.line(img, (0, ty), (w, ty), (200, 200, 0), 1)
                save_path = os.path.join(OUTPUT_DIR, f'comp_{img_file}')
                cv2.imwrite(save_path, img)
                found = True
                break
        if not found:
            results.append((img_file, '未检测到人物'))
    # 保存结果
    with open(RESULTS_FILE, 'w') as f:
        for img_file, result in results:
            f.write(f'{img_file}: {result}\n')
    print(f'构图分析完成，结果已保存到 {RESULTS_FILE}')
    print(f'可视化图片已保存到 {OUTPUT_DIR}/')

if __name__ == '__main__':
    main() 