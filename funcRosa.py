import numpy as np
import qimage2ndarray
from PyQt5.QtGui import *
import math

def edge_detection(image):
    image_array = qimage2ndarray.rgb_view(image)
    image_gray_array = np.dot(image_array,[0.2989, 0.5870, 0.1140])
    image_padding = np.pad(image_gray_array, 1, 'constant', constant_values=(0))

    kenel = np.array([[1 / 273, 4 / 273, 7 / 273, 4 / 273, 1 / 273],
                      [4 / 273, 16 / 273, 26 / 273, 16 / 273, 4 / 273],
                      [7 / 273, 26 / 273, 41 / 273, 26 / 273, 7 / 273],
                      [4 / 273, 16 / 273, 26 / 273, 16 / 273, 4 / 273],
                      [1 / 273, 4 / 273, 7 / 273, 4 / 273, 1 / 273]])

    image_2D = filtering(image_padding, kenel)

    lap_image = Laplacian(image_2D)

    image_final = qimage2ndarray.array2qimage(lap_image, normalize=False)
    return QPixmap.fromImage(image_final)

def getGKernel(shape, sigma):
    # a = shape[0] , b = shape[1] , (s = 2a+1, t = 2b+1)
    s = (shape[0] - 1) / 2
    t = (shape[1] - 1) / 2

    y, x = np.ogrid[-s:s + 1, -t:t + 1]
    gaus_kernel = np.exp(-(x * x + y * y)) / (2. * sigma * sigma)
    sum = gaus_kernel.sum()
    gaus_kernel /= sum
    return gaus_kernel

def filtering(pad_image, kernel):
    row, col = len(pad_image), len(pad_image[1])
    ksizeY, ksizeX = kernel.shape[0], kernel.shape[1]

    filtered_img = np.zeros((row, col), dtype=np.float32)
    for i in range(row-4):
        for j in range(col-4):
            filtered_img[i, j] = np.sum(np.multiply(kernel, pad_image[i:i + ksizeY, j:j + ksizeX]))  # filter * image
    return filtered_img


def Laplacian(gaus_array):

    d = gaus_array.shape
    n = d[0]
    m = d[1]
    lap_array = np.copy(gaus_array)

    for i in range(1, n - 1):
        for j in range(1, m - 1):
            lap = gaus_array[i - 1][j - 1] + gaus_array[i][j - 1] + gaus_array[i + 1][j - 1] + gaus_array[i - 1][j] + \
                  gaus_array[i][j] * (-8) + gaus_array[i + 1][j] + gaus_array[i - 1][j + 1] + gaus_array[i][j + 1] + \
                  gaus_array[i + 1][j + 1]
            lap_array[i][j] = lap

    return lap_array

###################################################################################################################

def hough(image):
    image_array = qimage2ndarray.rgb_view(image)
    image_gray_array = np.dot(image_array,[0.2989, 0.5870, 0.1140])
    image_padding = np.pad(image_gray_array, 1, 'constant', constant_values=(0))

    kenel = np.array([[1 / 273, 4 / 273, 7 / 273, 4 / 273, 1 / 273],
                      [4 / 273, 16 / 273, 26 / 273, 16 / 273, 4 / 273],
                      [7 / 273, 26 / 273, 41 / 273, 26 / 273, 7 / 273],
                      [4 / 273, 16 / 273, 26 / 273, 16 / 273, 4 / 273],
                      [1 / 273, 4 / 273, 7 / 273, 4 / 273, 1 / 273]])

    image_2D = filtering(image_padding, kenel)
    lap_image = Laplacian(image_2D)

    hough_image = build_hough_space_fom_image(lap_image)

    image_final = qimage2ndarray.array2qimage(hough_image, normalize=False)
    return QPixmap.fromImage(image_final)

def build_hough_space_fom_image(img, shape = (100, 300), val = 1):
    hough_space = np.zeros(shape)
    for i, row in enumerate(img):
        for j, pixel in enumerate(row):
            if pixel != val : continue
        hough_space = add_to_hough_space_polar((i,j), hough_space)
    return hough_space

def add_to_hough_space_polar(p, feature_space):
    space = np.linspace(0, math.pi, len(feature_space))
    d_max = len(feature_space[0]) / 2
    for i in range(len(space)):
        theta = space[i]
        d = int(p[0] * math.sin(theta) + p[1] * math.cos(theta)) + int(d_max)
        if (d >= d_max * 2):
            continue
        feature_space[i, d] += 1
    return feature_space
