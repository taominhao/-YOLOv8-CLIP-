import torch
from PIL import Image
import cv2

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
    score = int(60 + 40 * (similarity - 0.2) / 0.15)
    score = max(0, min(100, score))
    status = f'CLIP相关性分数：{similarity:.2f}'
    return score, status 