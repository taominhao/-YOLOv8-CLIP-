import cv2
import numpy as np

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
    penalty = sum(d < min_dist for d in [left, right, top, bottom]) * 10
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
        score = 60 + 15*on_third + 15*on_golden + min(10, area*100) - penalty
        result = f"{class_name}"
        color = (0, 255, 0) if on_third or on_golden else (0, 0, 255)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_img, result, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        for tx in [int(w*1/3), int(w*2/3)]:
            cv2.line(vis_img, (tx, 0), (tx, h), (200, 200, 0), 1)
        for ty in [int(h*1/3), int(h*2/3)]:
            cv2.line(vis_img, (0, ty), (w, ty), (200, 200, 0), 1)
        for gx in [int(w*0.382), int(w*0.618)]:
            for gy in [int(h*0.382), int(h*0.618)]:
                cv2.circle(vis_img, (gx, gy), 6, (255, 215, 0), -1)
        if score > best_score:
            best_score = score
            best_result = result
    if len(centers) > 1:
        centers = np.array(centers)
        spread = np.std(centers, axis=0).mean()
        best_score += min(10, spread / max(w, h) * 100)
        best_result += f"，多主体分布均匀度加分{min(10, spread / max(w, h) * 100):.1f}"
    if best_score == 0:
        best_score = 50
    return int(best_score), best_result, vis_img, main_classes 