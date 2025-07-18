import numpy as np

def light_shadow_score(gray):
    contrast = gray.std()
    score = int(min(100, contrast*2))
    status = '丰富' if contrast > 40 else ('一般' if contrast > 25 else '较弱')
    return score, status, contrast 