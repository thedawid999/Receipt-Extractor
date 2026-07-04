import cv2
import numpy as np 
from deskew import determine_skew

# adds grayscale and rotates the image if skewed
def preprocess(path: str) -> np.ndarray:
    img = cv2.imread(path)

    # add grayscale (black-and-white)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # rotate
    angle = determine_skew(grayed)
    rotated = rotate(angle, grayed)

    return rotated

def rotate(angle:float, img: np.ndarray):
    (h, w) = img.shape
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated
