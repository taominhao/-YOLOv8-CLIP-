import cv2
import numpy as np

def focus_score(gray):
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = min(100, max(0, (lap_var-50)/1.5))
    status = '清晰' if lap_var > 200 else ('一般' if lap_var > 100 else '模糊')
    return int(score), status, lap_var 