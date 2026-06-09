import cv2 as cv2
import numpy as np 
from matplotlib import pyplot as plt 
import os

path = "samples/original"
new_path = "samples/preprocessed"

for file in os.listdir(path):
    file_path = os.path.join(path, file)

    img = cv2.imread(file_path)
    # convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # add grayscale (black-and-white)
    grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # remove noise
    denoised = cv2.bilateralFilter(grayed, d=3, sigmaColor=40, sigmaSpace=40)
    # binarise (increases contrast additionaly)
    binarised = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 6)





    # CREATED TO FIND BEST PARAMETER VALUES FOR PREPROCESSING
    #titles = ["original", "grayscaled", "denoised", "binarised"]
    #images = [img_rgb, grayed, denoised, binarised]

    #plt.figure(figsize=(8,18))
    #for i in range(4):
    #    plt.subplot(2,5,i+1)
    #    if i == 0:
    #        plt.imshow(images[i])
    #    else:
    #        plt.imshow(images[i], cmap="gray")
    #    plt.title(titles[i])
    #    plt.axis("off")

    #    new_filename = os.path.join(new_path, file)
    #plt.savefig(new_filename, dpi=300, bbox_inches="tight")
