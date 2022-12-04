import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

# Function: Read images as grayscale images
def read_images(path, size = (256, 256)):
    I_plains = os.listdir(path)
    N_files = len(I_plains)
    kI = []
    str_Fnames = []

    for i in range(0, N_files):
        str = I_plains[i]
        str_Fnames.append(str)
        Ip = cv2.imread(path + '\\' + str, 0)
        Ip_resized = cv2.resize(Ip, size, interpolation = cv2.INTER_AREA)
        kI.append(Ip_resized)
    return kI, str_Fnames


# Function: Show images as grayscale images
def show_images(kI, str_Fnames, size = (10, 10), rows = 3, cols = 3):
    fig = plt.figure(figsize = size)

    for i in range(len(kI)):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(kI[i], cmap='gray', vmin=0, vmax=255)
        plt.title(str_Fnames[i])
    plt.show()

def save_images(folder_path):
    pass