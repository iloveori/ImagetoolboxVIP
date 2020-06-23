#해리스 코너 검출이용해서 코너를 검출
#import문
import numpy
import cv2
import qimage2ndarray
import math
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *


#이미지 로드- opencv->numpy 할때만 사용


def cornerdetection (image):
    #이미지 불러오기


    image_array = qimage2ndarray.rgb_view(image)
    #회색 처리
    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    image_gray = numpy.float32(image_gray)
    #?
    
    #3by3 matrix만들어서 이미지의 3by3 matrix와 내적.
    #이때 이미지는 미분먼저 한 다음 내적 해야된다.. 

    

    #x축으로 미분
    image_x = cv2.Sobel(image_gray, cv2.CV_64F ,1, 0 )
    #y축으로 미분
    image_y = cv2.Sobel(image_gray, cv2.CV_64F ,0, 1 )
    
    #내적(곱)
    _xx = image_x*image_x
    _xy = image_x*image_y
    _yy = image_y*image_y

    #이미지의 가로세로 길이
    height,width = image_array.shape[0:2]
    window_size = 3
    offset = int(window_size/2)#1
    
    R=numpy.zeros(image_gray.shape)

    #한 픽셀씩 이동하면서 계산
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            window_xx = _xx[y-offset:y+offset+1, x-offset:x+offset+1]
            window_xy = _xy[y-offset:y+offset+1, x-offset:x+offset+1]
            window_yy = _yy[y-offset:y+offset+1, x-offset:x+offset+1]

            xx = window_xx.sum()
            xy = window_xy.sum()
            yy = window_yy.sum()

            det=xx*yy-xy**2
            trace = xx+yy
            #R 식, k=0.04 by harris
            R[y,x] = det -0.0435*(trace**2)
            #R=det-0.04*(trace**2)
            
    R = cv2.normalize(R,0,0.0,1.0,cv2.NORM_MINMAX)  

    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            #왜 하필 0.04일까?
            if R[y,x]>0.4:
                #해당 픽셀에 채널값을 준다.픽셀값 설정.
                image_array.itemset((y,x,0),0)
                image_array.itemset((y,x,1),0)
                image_array.itemset((y,x,2),255)
                
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    image_array = qimage2ndarray.array2qimage(image_array, normalize=False)
    qPixmapVar3 = QPixmap.fromImage(image_array)
    return qPixmapVar3
