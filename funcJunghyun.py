import qimage2ndarray
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *

# cv2 import 된거 주석처리하고 실행하기


def get_gaussian_kernel(kernel_size, sigma):
    """
    2차원 가우시안 필터 생성해서 반환

    :param kernel_size: N
    :param sigma: sigma
    :return: xN 크기의 가우시안 필터 생성
    """
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return kernel / np.sum(kernel)


def gaussian_filtering(image, ksize, sigma):
    """
    가우시안 필터링 결과 이미지 반환

    :param image: 입력 이미지
    :param ksize: 필터 사이즈
    :param sigma: sigma
    :return: 필터링된 결과 이미지
    """
    kernel = get_gaussian_kernel(ksize, sigma)

    # convolution operation
    [image_row, image_col, _] = image.shape
    [kernel_row, kernel_col] = kernel.shape

    output = np.zeros(image.shape)
    pad_h = (kernel_row - 1) // 2
    pad_w = (kernel_col - 1) // 2

    padded_image = np.zeros((image_row + (2 * pad_h), image_col + (2 * pad_w)))
    padded_image[pad_h:padded_image.shape[0] - pad_h, pad_w:padded_image.shape[1] - pad_w] = image[:, :, 0]

    for r in range(image_row):
        for c in range(image_col):
            output[r, c] = np.sum(kernel * padded_image[r:r + kernel_row, c:c + kernel_col])

    return output


def laplacian_filtering(image):
    """
    라플라시안 필터링을 통해 엣지 검출

    :param image: 입력 이미지
    :return: 필터링 결과
    """
    kernel = [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ]

    # convolution operation
    [image_row, image_col, _] = image.shape
    kernel_row, kernel_col = 3, 3

    output = np.zeros(image.shape)
    pad_h = (kernel_row - 1) // 2
    pad_w = (kernel_col - 1) // 2

    padded_image = np.full((image_row + (2 * pad_h), image_col + (2 * pad_w)), 128)
    padded_image[pad_h:padded_image.shape[0] - pad_h, pad_w:padded_image.shape[1] - pad_w] = image[:, :, 0]

    for r in range(image_row):
        for c in range(image_col):
            output[r, c] = np.sum(kernel * padded_image[r:r + kernel_row, c:c + kernel_col])

    return output


def gray_scale(image):
    params = [0.299, 0.587, 0.144]

    # image는 현재 Q-image 이므로 numpy로 변환한다. (512 x 512 x 3)
    image_array = qimage2ndarray.rgb_view(image)

    # 변환 결과 이미지 (512 x 512 x 1)
    result = np.zeros([len(image_array), len(image_array[0]), 1])

    for r in range(len(image_array)):
        for c in range(len(image_array[0])):
            value = image_array[r][c]
            dot = sum(value * params)
            result[r][c][0] = dot

    return result


def EdgeDetection(image):
    image = gray_scale(image)
    image = gaussian_filtering(image, 5, 1)
    image = laplacian_filtering(image)
    image = qimage2ndarray.array2qimage(image, normalize=False)
    image = QPixmap.fromImage(image)

    return image
