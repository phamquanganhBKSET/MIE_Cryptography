import cv2
import os
from matplotlib import pyplot as plt

# Function: Read images as grayscale images
# path: Directory
# size: Size of images
# OS: Name of OS
# Return: List of image matrices (list of numpy arrays)
def read_images(path, size = (256, 256), OS = 'Windows'):
    I_plains = os.listdir(path)
    N_files = len(I_plains)
    kI = []
    str_Fnames = []

    for i in range(0, N_files):
        str = I_plains[i]
        str_Fnames.append(str)
        Ip = cv2.imread(path + '\\' + str if OS == 'Windows' else path + '/' + str, 0)
        Ip_resized = cv2.resize(Ip, size, interpolation = cv2.INTER_AREA)
        kI.append(Ip_resized)
    return kI, str_Fnames


# Function: Show images as grayscale images
# kI: List of image matrices
# str_Fnames: List of image names
# size: Size of a image window
# rows: Number of rows presented
# cols: Number of columns presented
def show_images(kI, str_Fnames, size = (10, 10), rows = 3, cols = 3):
    fig = plt.figure(figsize = size)

    for i in range(len(kI)):
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(kI[i], cmap='gray', vmin=0, vmax=255)
        plt.title(str_Fnames[i])
    fig.suptitle('Plain images', size = 16)
    fig.tight_layout(pad=1.0)
    plt.show()

# Function: Save image into directory
def save_images(kC, folder_path, str_Fnames):
    for i in range(len(str_Fnames)):
        cv2.imwrite(folder_path + str_Fnames[i], kC[i])
    