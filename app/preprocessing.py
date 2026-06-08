import cv2 as cv2
import numpy as np 
from matplotlib import pyplot as plt 

img = cv2.imread("receipt.jpg")
# convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# add grayscale (black-and-white)
grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# remove noise (add median blur)
denoised = cv2.medianBlur(grayed, 3)
# binarise (increases contrast additionaly)
binarised = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

titles = ["original", "grayscaled", "denoised", "binarised"]
images = [img_rgb, grayed, denoised, binarised]

plt.figure(figsize=(12,8))
for i in range(4):
    plt.subplot(2,2,i+1)
    if i == 0:
        plt.imshow(images[i])
    else:
        plt.imshow(images[i], cmap="gray")
    plt.title(titles[i])
    plt.axis("off")

plt.show()