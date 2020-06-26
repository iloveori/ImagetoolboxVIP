from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
import qimage2ndarray
import numpy as np






#메디안필터
def median_f(data, kernel_size):
	temp=[]
	indexer= kernel_size//2
	data_final =[]
	data_final = np.zeros((len(data),len(data[0])))
	for i in range(len(data[0])):
		for j in range(len(data[0])):
			for z in range(kernel_size):
				if i+z-indexer < 0 or i+z-indexer > len(data) -1:
					for c in range(kernel_size):
						temp.append(0)
				else:
					if j+z-indexer < 0 or j +indexer > len(data[0]) -1:
						temp.append(0)
					else:
						for k in range(kernel_size):
							temp.append(data[i+z-indexer][j+k-indexer])
			temp.sort()
			data_final[i][j] = temp[len(temp)//2]
			temp=[]
	return data_final


        
# 라플라시안 필터
def Laplacian(gaus_array):
    
    matrix_n=gaus_array.shape[0] 
    matrix_m=gaus_array.shape[1]
    lap_array=np.copy(gaus_array)

    for j in range(1,matrix_n-1):
        for i in range(1,matrix_m-1):
            lap_array[i][j]=gaus_array[i-1][j-1]+gaus_array[i][j-1]+gaus_array[i+1][j-1]+gaus_array[i-1][j]+gaus_array[i][j]*(-8)+gaus_array[i+1][j]+gaus_array[i][j+1]+gaus_array[i-1][j+1]+gaus_array[i+1][j+1]


            
    return lap_array

#제로크로싱
def zeroCrossing(lap_arr):
    width,height = lap_arr.shape
    Z=[]
    Z=np.zeros(lap_arr.shape)
    
    for y in range(1,width-2):
        for x in range(1,height-2):
            neighbors = [lap_arr[y-1,x],lap_arr[y+1,x],lap_arr[y,x-1],lap_arr[y,x+1],lap_arr[y-1,x-1],lap_arr[y-1,x+1],lap_arr[y+1,x-1],lap_arr[y+1,x+1]]
            mValue = min(neighbors) 
            if SGN(lap_arr[y,x]) != SGN(mValue): 
                Z[y,x] = 255
            else:
                Z[y,x] =0
    
    return Z


def SGN(x):
    if x>=0.5:
        sign=1       
    else:        
        sign=-1
    return sign


#엣지 검출
def Edge_detect(image):
    image_array = qimage2ndarray.rgb_view(image)
    gray_coeff=[0.2989,0.5870,0.1140]
    gray_array=np.dot(image_array,gray_coeff)
    
    median_array1=median_f(gray_array,3)
    
    # median_array2=median_f(median_array1,3)
    # median_array3=median_f(median_array2,3)
    # median_array4=median_f(median_array3,3)
    # median_array5=median_f(median_array4,3)
    # median_array6=median_f(median_array5,3)
    # median_array7=median_f(median_array6,3)
    # median_array8=median_f(median_array7,3)
    # median_array=median_f(median_array8,3)

    laplacian_array=Laplacian(median_array1)

    zerocrossing_array=zeroCrossing(laplacian_array)

    image=qimage2ndarray.array2qimage(zerocrossing_array, normalize=False)
    qPixmapVar = QPixmap.fromImage(image) 
    return qPixmapVar


#코너 검출
def Corner(image):
    
    image_array = qimage2ndarray.rgb_view(image)
    gray_coeff=[0.2989,0.5870,0.1140]
    gray_array=np.dot(image_array,gray_coeff)
    matrix_n=gray_array.shape[0] 
    matrix_m=gray_array.shape[1]
    corner_array=np.copy(gray_array)

    for j in range(1,matrix_n-1):
        for i in range(1,matrix_m-1):
            Det=gray_array[i-1][j-1]*gray_array[i][j]*gray_array[i+1][j+1] + gray_array[i-1][j]*gray_array[i][j+1]*gray_array[i+1][j+1] + gray_array[i-1][j+1]*gray_array[i][j-1]*gray_array[i+1][j] - ( gray_array[i+1][j-1]*gray_array[i][j]*gray_array[i-1][j+1] + gray_array[i-1][j]*gray_array[i][j-1]*gray_array[i+1][j+1] + gray_array[i-1][j-1]*gray_array[i][j+1]*gray_array[i+1][j] )
            
            trace=gray_array[i-1][j-1] + gray_array[i][j] + gray_array[i+1][j+1]
            

            harris=Det-0.04*trace*trace

            if (harris>0.1):
                corner_array[i,j]=[255,0]


            
    image=qimage2ndarray.array2qimage(corner_array, normalize=False)
    qPixmapVar = QPixmap.fromImage(image) 
    return qPixmapVar
    




    
