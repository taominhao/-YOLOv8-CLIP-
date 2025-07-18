import cv2
import numpy as np

def color_score(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([img_hsv], [0], None, [36], [0, 180])
    main_color_ratio = h_hist.max() / h_hist.sum()
    score = int(60 + 40 * main_color_ratio)
    status = '统一' if main_color_ratio > 0.4 else '一般'
    return score, status, main_color_ratio 