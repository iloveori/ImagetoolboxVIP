import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
import qimage2ndarray
import numpy as np
import cv2

def edge_detection(image):
    # qImage to numpy
    image_array = qimage2ndarray.rgb_view(image)

    # convert to grayscale
    image_array = 0.299 * image_array[:, :, 0] + 0.587 * image_array[:, :, 1] + 0.114 * image_array[:, :, 2]



    # 가우시안 필터

    height, width = image_array.shape
    padding_array = [[0 for col in range(height+2)] for row in range(width+2)]  # 이차원 배열 선언

    # padding
    for i in range(height+2):
        for j in range(width+2):
            if (i==0) or (i==height+1) or (j==0) or (j==width+1):
                padding_array[i][j] = 128;
            else:
                padding_array[i][j] = image_array[i-1][j-1];

    GaussianFilter = [[1,2,1],[2,4,2],[1,2,1]]
    filter_sum = 16

    for arr_row in range(0, height):
        for arr_col in range(0, width):
            product = 0

            for ft_row in range(0, 3):
                for ft_col in range(0, 3):
                    product += padding_array[arr_row+ft_row][arr_col+ft_col] * GaussianFilter[ft_row][ft_col]
            product /= filter_sum
            image_array[arr_row][arr_col] = product




    # Laplacian 필터

    # padding
    for i in range(height + 2):
        for j in range(width + 2):
            if (i == 0) or (i == height + 1) or (j == 0) or (j == width + 1):
                padding_array[i][j] = 128;
            else:
                padding_array[i][j] = image_array[i - 1][j - 1];

    LaplacianFilter = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]

    for arr_row in range(0, height):
        for arr_col in range(0, width):
            product = 0

            for ft_row in range(0, 3):
                for ft_col in range(0, 3):
                    product += padding_array[arr_row+ft_row][arr_col+ft_col] * LaplacianFilter[ft_row][ft_col]
            image_array[arr_row][arr_col] = product
            



    # zero crossing (에러 원인을 모르겠습니다ㅠㅠ)

    # zerocross_array = [[0 for col in range(height)] for row in range(width)]  # 이차원 배열 생성
    #
    # for row in range(0, height):
    #     for col in range(1, width-1):
    #         if (image_array[row][col-1] * image_array[row][col+1]) < 0:
    #             zerocross_array[row][col] = 128;
    # for row in range(1, height-1):
    #     for col in range(0, width):
    #         if (image_array[row-1][col] * image_array[row+1][col]) < 0:
    #             zerocross_array[row][col] = 128;




    # numpy to qImage
    image = qimage2ndarray.array2qimage(image_array, normalize=False)
    # QImage to QPixmap
    qPixmapVar = QPixmap.fromImage(image)


    return qPixmapVar


