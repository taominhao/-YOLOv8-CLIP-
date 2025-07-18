import cv2
import numpy as np
from imquality import brisque
from skimage import img_as_ubyte

def noise_score(img):
    scores = []
    for i in range(3):
        channel = img[:, :, i]
        high_freq = channel - cv2.medianBlur(channel, 5)
        scores.append(np.std(high_freq))
    noise = np.mean(scores)
    score = max(0, 100 - (noise-8)*4)
    status = '优秀' if noise < 8 else ('一般' if noise < 15 else '较多')
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    brisque_score = brisque.score(img_as_ubyte(img_rgb))
    return int(score), status, noise, brisque_score 