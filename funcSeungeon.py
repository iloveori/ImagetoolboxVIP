from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
import qimage2ndarray
import numpy as np
import math


def Edge_Detection(image):
    image_arr = qimage2ndarray.rgb_view(image)

    #Grayscale
    gray_arr = [0.2890, 0.5870, 0.1140]
    result = np.dot(image_arr, gray_arr)


    for i in range(1,10):
        #Padding
        image_pad = np.pad(result, 2, mode='constant', constant_values=0)
    
        #Smoothing
        kenel = np.array([[1/25,1/25,1/25,1/25,1/25],
                          [1/25,1/25,1/25,1/25,1/25],
                          [1/25,1/25,1/25,1/25,1/25],
                          [1/25,1/25,1/25,1/25,1/25],
                          [1/25,1/25,1/25,1/25,1/25]])



        a = image_pad.shape[0]-kenel.shape[0] + 1
        b = image_pad.shape[1]-kenel.shape[1] + 1

    
        result2 = []
        for x in range(a):
            for y in range(b):
                result1 = image_pad[ x : x + kenel.shape[0], y : y + kenel.shape[1] ] * kenel
                result2.append(np.sum(result1))
            
        result = np.array(result2).reshape(a,b)


    #Padding
    image_pad = np.pad(result, 2, mode='constant', constant_values=0)

    #Laplacian Filtering
    kenel = np.array([[0,0,1,0,0],
                     [0,1,2,1,0],
                     [1,2,-16,2,1],
                     [0,1,2,1,0],
                     [0,0,1,0,0]])

    a = image_pad.shape[0]-kenel.shape[0] + 1
    b = image_pad.shape[1]-kenel.shape[1] + 1
    result2 = []
    for x in range(a):
        for y in range(b):
            result1 = image_pad[ x : x + kenel.shape[0], y : y + kenel.shape[1] ] * kenel
            result2.append(np.sum(result1))
    result = np.array(result2).reshape(a,b)


    #Padding
    image_pad = np.pad(result, 1, mode='constant', constant_values=0)


    #Zerocrossing
    a = image_pad.shape[0]
    b = image_pad.shape[1]
    result1 =[]

    for x in range(a-2):
        for y in range(b-2):
            neighbors = [image_pad[x-1,y],image_pad[x+1,y],image_pad[x,y-1],image_pad[x,y+1],image_pad[x-1,y-1],image_pad[x-1,y+1],image_pad[x+1,y-1],image_pad[x+1,y+1]]
            mValue = min(neighbors)
            if SGN(image_pad[x,y]) != SGN(mValue):
                result1.append(255)
            else:
                result1.append(0)
    result = np.array(result1).reshape(a-2,b-2)

    image_after = qimage2ndarray.array2qimage(result, normalize=False)

    return QPixmap.fromImage(image_after)


def SGN(x):
    if x>0.01:
        sign=1
    else:
        sign=0
    return sign


def Corner_Detection(image):
    image_arr = qimage2ndarray.rgb_view(image)

    #Grayscale
    gray_arr = [0.2890, 0.5870, 0.1140]
    result = np.dot(image_arr, gray_arr)
        

    #Padding
    image_pad = np.pad(result, 1, mode='constant', constant_values=0)


    #Get Gradient

    x_kenel = np.array([[-1,-2,-1],
                       [0,0,0],
                       [1,2,1]])

    a = image_pad.shape[0]-x_kenel.shape[0] + 1
    b = image_pad.shape[1]-x_kenel.shape[1] + 1

    result2 = []

    for x in range(a):
            for y in range(b):
                result1 = image_pad[ x : x + x_kenel.shape[0], y : y + x_kenel.shape[1] ] * x_kenel
                result2.append(np.sum(result1))
            
    x_gradient = np.array(result2).reshape(a,b)

    y_kenel = np.array([[-1,0,1],
                       [-2,0,2],
                       [-1,0,1]])

    a = image_pad.shape[0]-y_kenel.shape[0] + 1
    b = image_pad.shape[1]-y_kenel.shape[1] + 1

    result2 = []

    for x in range(a):
            for y in range(b):
                result1 = image_pad[ x : x + y_kenel.shape[0], y : y + y_kenel.shape[1] ] * y_kenel
                result2.append(np.sum(result1))

    y_gradient = np.array(result2).reshape(a,b)

    
    result2 = []
    for x in range(512):
        for y in range(b):
            result1 = math.sqrt((x_gradient[x,y])*(x_gradient[x,y])+(y_gradient[x,y])*(y_gradient[x,y]))
            result2.append(np.result1)
            

    



    image_after = qimage2ndarray.array2qimage(y_gradient, normalize=False)

    return QPixmap.fromImage(image_after)



    


    

    
    


