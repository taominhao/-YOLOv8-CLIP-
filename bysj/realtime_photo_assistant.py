import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from collections import Counter
import open_clip
import torch
from PIL import Image
import os
from transformers import AutoImageProcessor, AutoModelForImageClassification
from imquality import brisque
from skimage import img_as_ubyte

# 曝光打分参数
UNDER_EXPOSED_THRESHOLD = 60
OVER_EXPOSED_THRESHOLD = 190

# 三分法线比例
THIRD_RATIO = 1/3

# YOLO模型
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_clip():
    import clip
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, None  # tokenizer 用 clip.tokenize 代替



def exposure_score(gray):
    mean_brightness = np.mean(gray)
    # 理想曝光为120~150，满分100，偏离越大分数越低
    ideal = 135
    score = max(0, 100 - abs(mean_brightness - ideal) * 1.2)
    if mean_brightness < UNDER_EXPOSED_THRESHOLD:
        status = '欠曝'
    elif mean_brightness > OVER_EXPOSED_THRESHOLD:
        status = '过曝'
    else:
        status = '正常'
    return int(score), status, mean_brightness

def focus_score(gray):
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 经验阈值：>200清晰，<50模糊
    score = min(100, max(0, (lap_var-50)/1.5))
    status = '清晰' if lap_var > 200 else ('一般' if lap_var > 100 else '模糊')
    return int(score), status, lap_var

def noise_score(img):
    # img为BGR格式
    scores = []
    for i in range(3):  # 分别处理B、G、R通道
        channel = img[:, :, i]
        high_freq = channel - cv2.medianBlur(channel, 5)
        scores.append(np.std(high_freq))
    noise = np.mean(scores)
    # 归一化到0~100分（经验值，8以下优秀，25以上较差）
    score = max(0, 100 - (noise-8)*4)
    status = '优秀' if noise < 8 else ('一般' if noise < 15 else '较多')
    # BRISQUE分数
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    brisque_score = brisque.score(img_as_ubyte(img_rgb))
    return int(score), status, noise, brisque_score

def is_on_third_line(box, img_shape):
    (h, w) = img_shape[:2]
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    thirds_x = [w * 1/3, w * 2/3]
    thirds_y = [h * 1/3, h * 2/3]
    tol_x = w * 0.08
    tol_y = h * 0.08
    on_x = any(abs(cx - tx) < tol_x for tx in thirds_x)
    on_y = any(abs(cy - ty) < tol_y for ty in thirds_y)
    return on_x or on_y

def is_near_golden_point(cx, cy, w, h, tol=0.08):
    golden_x = [w * 0.382, w * 0.618]
    golden_y = [h * 0.382, h * 0.618]
    return any(abs(cx - gx) < w * tol for gx in golden_x) and any(abs(cy - gy) < h * tol for gy in golden_y)

def border_penalty(x1, y1, x2, y2, w, h, min_dist=0.05):
    left = x1 / w
    right = (w - x2) / w
    top = y1 / h
    bottom = (h - y2) / h
    penalty = sum(d < min_dist for d in [left, right, top, bottom]) * 10  # 每靠近一边扣10分
    return penalty

def composition_score(img, model):
    h, w = img.shape[:2]
    pred = model(img)[0]
    best_score = 0
    best_result = '未检测到主体'
    vis_img = img.copy()
    main_classes = []
    centers = []
    for box, cls in zip(pred.boxes.xyxy.cpu().numpy(), pred.boxes.cls.cpu().numpy()):
        class_name = model.model.names[int(cls)]
        main_classes.append(class_name)
        x1, y1, x2, y2 = map(int, box)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        centers.append((cx, cy))
        on_third = is_on_third_line((x1, y1, x2, y2), img.shape)
        on_golden = is_near_golden_point(cx, cy, w, h)
        area = (x2-x1)*(y2-y1)/(w*h)
        penalty = border_penalty(x1, y1, x2, y2, w, h)
        # 综合得分：三分法+黄金分割+面积+边界惩罚
        score = 60 + 15*on_third + 15*on_golden + min(10, area*100) - penalty
        result = f"{class_name}"  # 只显示类别名
        color = (0, 255, 0) if on_third or on_golden else (0, 0, 255)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_img, result, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        for tx in [int(w*1/3), int(w*2/3)]:
            cv2.line(vis_img, (tx, 0), (tx, h), (200, 200, 0), 1)
        for ty in [int(h*1/3), int(h*2/3)]:
            cv2.line(vis_img, (0, ty), (w, ty), (200, 200, 0), 1)
        # 黄金分割点可视化
        for gx in [int(w*0.382), int(w*0.618)]:
            for gy in [int(h*0.382), int(h*0.618)]:
                cv2.circle(vis_img, (gx, gy), 6, (255, 215, 0), -1)
        if score > best_score:
            best_score = score
            best_result = result
    # 多主体分布分析（如有多个主体，分析中心点分布）
    if len(centers) > 1:
        import numpy as np
        centers = np.array(centers)
        spread = np.std(centers, axis=0).mean()
        # 分布越均匀，得分越高
        best_score += min(10, spread / max(w, h) * 100)
        best_result += f"，多主体分布均匀度加分{min(10, spread / max(w, h) * 100):.1f}"
    if best_score == 0:
        best_score = 50
    return int(best_score), best_result, vis_img, main_classes

def color_score(img):
    # 色调统一：主色占比越高分越高
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([img_hsv], [0], None, [36], [0, 180])
    main_color_ratio = h_hist.max() / h_hist.sum()
    score = int(60 + 40 * main_color_ratio)
    status = '统一' if main_color_ratio > 0.4 else '一般'
    return score, status, main_color_ratio

def light_shadow_score(gray):
    # 光影层次：对比度越高分越高
    contrast = gray.std()
    score = int(min(100, contrast*2))
    status = '丰富' if contrast > 40 else ('一般' if contrast > 25 else '较弱')
    return score, status, contrast

def content_score_clip(img, theme, clip_model, clip_preprocess, _):
    import clip
    device = 'cpu'
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    image_input = clip_preprocess(pil_img).unsqueeze(0).to(device)
    text_input = clip.tokenize([theme]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()
    # CLIP相似度一般在0.2~0.35之间，线性映射到60~100分
    score = int(60 + 40 * (similarity - 0.2) / 0.15)
    score = max(0, min(100, score))
    status = f'CLIP相关性分数：{similarity:.2f}'
    return score, status

def main():
    st.title('多维度实时照片打分系统（支持主题表达）')
    st.write('上传照片，输入主题，系统将自动从7个维度分析并打分，给出优化建议。')
    uploaded_file = st.file_uploader('请上传一张照片', type=['jpg', 'jpeg', 'png'])
    theme = st.text_input('请输入本照片的主题词/短语（如“城市夜景”、“自然风光”、“现代建筑”等）', value='风景')
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        img = cv2.imread(tfile.name)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption='原图', use_container_width=True)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 1. 曝光
        exp_score, exp_status, mean_brightness = exposure_score(gray)
        # 2. 对焦
        foc_score, foc_status, lap_var = focus_score(gray)
        # 3. 噪点
        noi_score, noi_status, noise, brisque_score = noise_score(img)
        # 4. 构图
        model = load_yolo()
        comp_score, comp_result, vis_img, main_classes = composition_score(img, model)
        # 5. 色调
        col_score, col_status, main_color_ratio = color_score(img)
        # 6. 光影
        ls_score, ls_status, contrast = light_shadow_score(gray)
        # 7. 内容表达
        clip_model, clip_preprocess, clip_tokenizer = load_clip()
        cont_score, cont_status = content_score_clip(img, theme, clip_model, clip_preprocess, clip_tokenizer)
        # 展示各项得分
        st.subheader('各维度打分')
        st.write(f'曝光：{exp_score}/100（{exp_status}，平均亮度{mean_brightness:.2f}）')
        st.write(f'对焦：{foc_score}/100（{foc_status}，Laplacian方差{lap_var:.2f}）')
        st.write(f'噪点：{noi_score}/100（{noi_status}，高频能量{noise:.2f}，BRISQUE分数{brisque_score:.2f}）')
        st.write(f'构图：{comp_score}/100（{comp_result}）')
        st.write(f'色调：{col_score}/100（{col_status}，主色占比{main_color_ratio:.2f}）')
        st.write(f'光影：{ls_score}/100（{ls_status}，对比度{contrast:.2f}）')
        st.write(f'内容表达：{cont_score}/100（{cont_status}，主题：{theme}）')
        st.image(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB), caption='构图分析可视化', use_container_width=True)
        # 综合得分
        total_score = int(np.mean([exp_score, foc_score, noi_score, comp_score, col_score, ls_score, cont_score]))
        st.subheader('综合得分')
        st.write(f'综合得分：{total_score}/100')
        # 优化建议
        st.subheader('优化建议')
        advice = []
        if exp_status == '欠曝':
            advice.append('建议提高曝光或补光。')
        elif exp_status == '过曝':
            advice.append('建议降低曝光或减少强光。')
        else:
            advice.append('曝光良好。')
        if foc_score < 70:
            advice.append('建议对焦更精准，避免画面模糊。')
        else:
            advice.append('对焦良好。')
        if noi_score < 70:
            advice.append('建议降低ISO或用降噪工具。')
        else:
            advice.append('噪点控制良好。')
        if comp_score < 70:
            advice.append('建议主体靠近三分线或黄金分割点，提升画面美感。')
        else:
            advice.append('构图较好。')
        if col_score < 70:
            advice.append('建议统一色调，避免杂色。')
        else:
            advice.append('色调和谐。')
        if ls_score < 70:
            advice.append('建议增强光影层次，提升立体感。')
        else:
            advice.append('光影层次丰富。')
        if cont_score < 70:
            advice.append('建议突出主题，增强内容表达。')
        else:
            advice.append('内容表达贴合主题。')
        st.write(' '.join(advice))

if __name__ == '__main__':
    main() 