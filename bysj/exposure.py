import numpy as np

UNDER_EXPOSED_THRESHOLD = 60
OVER_EXPOSED_THRESHOLD = 190

def exposure_score(gray):
    mean_brightness = np.mean(gray)
    ideal = 135
    score = max(0, 100 - abs(mean_brightness - ideal) * 1.2)
    if mean_brightness < UNDER_EXPOSED_THRESHOLD:
        status = '欠曝'
    elif mean_brightness > OVER_EXPOSED_THRESHOLD:
        status = '过曝'
    else:
        status = '正常'
    return int(score), status, mean_brightness 