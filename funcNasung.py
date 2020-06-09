from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtGui import *
import qimage2ndarray
import numpy as np
import math
import cv2

# 그레이 스케일 
def Gray_scale(image_arr):
   
    arr=[0.299, 0.587, 0.114]  #rgb를 Grayscale로 변환하는 공식 
    gray_arr=image_arr.dot(arr)
    return gray_arr

#패딩 
def padding(gray_arr):

    image_pad = np.pad(gray_arr, 1, mode='constant', constant_values=0)

    return image_pad

# 가우시안 필터 
def Gaussian_filter(gray_arr):

    dims=gray_arr.shape
    n=dims[0]; m=dims[1] 
    gaus_arr=np.copy(gray_arr)

    for i in range(n):
        for j in range(m):
            gaus_arr[i][j]=0
    
    for i in range(1,n-1):
        for j in range(1,m-1):
            gaus_arr[i,j]+=(gray_arr[i-1,j]+gray_arr[i+1,j]+gray_arr[i,j-1]+gray_arr[i,j+1])*0.5
            gaus_arr[i,j]+=(gray_arr[i-1,j-1]+gray_arr[i-1,j+1]+gray_arr[i+1,j-1]+gray_arr[i+1,j+1])*0.25     
            gaus_arr[i,j]+=gray_arr[i][j]       
    gaus_arr/=4.
    return gaus_arr

#라플라시안 필터  
def Laplacian(gaus_arr):
    # 커널 형식 [0,1,0],[1,-4,1],[0,1,0]
    
    dims=gaus_arr.shape
    n=dims[0]; m=dims[1]
    lap_arr=np.copy(gaus_arr)

    for i in range(1,n-1):
        for j in range(1,m-1):
            lap=gaus_arr[i-1][j-1]+gaus_arr[i][j-1]+gaus_arr[i+1][j-1]+gaus_arr[i-1][j]+gaus_arr[i][j]*(-8)+gaus_arr[i+1][j]+gaus_arr[i-1][j+1]+gaus_arr[i][j+1]+gaus_arr[i+1][j+1]            
           
            lap_arr[i][j]=lap
    
    return lap_arr

#zero-crossing #좌우상하의 각 곱이 음수인 경우 zero-crossing 함 

def zerocrossing(lap_arr):
    dims=lap_arr.shape
    n=dims[0]; m=dims[1]
    zero_arr=np.copy(lap_arr)
    
    for i in range(n):
        for j in range(m):
            zero_arr[i][j]=0

    for i in range(1,n-1):
        for j in range(1,m-1):
            if(lap_arr[i][j]>=0): #양수인 경우
                    
                if(((lap_arr[i-1][j]*lap_arr[i+1][j])<0) or ((lap_arr[i][j-1]*lap_arr[i][j+1])<0)): #화소의 좌우 부분의 곱이 음수인 경우 or
                    zero_arr[i][j]=255 #2차 미분값이 0인 경우 화이트 출력
                
                else:
                    zero_arr[i][j]=0 #2차 미분값이 0이 아닌 경우 블랙 출력
            
    return zero_arr

#허프변환 (영상의 한 점을 거리와 각도 공간으로 바꾸는 과정)

def hough(image_arr, zero_arr):

    kThreshHoldLine=110 #직선을 찾기 위한 임곗값으로, 직선을 구성하는 점의 최소한 voting개수
    dims=zero_arr.shape 
    n=dims[0]; m=dims[1] # n*m배열 
                   
    angle=0
    rho=0 
    Range=int(math.sqrt((n*n)+(m*m)))  #이미지의 최대 대각선 길이 (a제곱+b제곱=대각선의 제곱)
    
    angle_list=[0,] #임곗값을 넘는 p와 angle값 저장하기 위함 
    rho_list=[0,]
    Hough = [[0 for col in range(Range*2)] for row in range(180)] #voting 배열 초기화 
    print(zero_arr)
    for i in range(0,n-1): #이미지의 높이
        for j in range(0,m-1): #이미지의 너비
            if(zero_arr[i][j]==255): #엣지인 경우
                for angle in range(0,180-1): #angle의 범위는 0~180도에서 1단위로 설정
                    rho= int(np.sin(angle*(np.pi / 180))*i + np.cos(angle*(np.pi / 180))*j)  #직선의 방정식 x=i, y=j 
                                                            #angle은 원점에서 직선에 수직선을 그렸을 때 y축과 이루는 각도의 크기 
                                                            #rho는 원점에서 직선까지의 수직의 거리 
            
                    Hough[angle][rho]+=1 #직선을 구성할 가능성이 있을 경우, 1씩 누적하여 투표 
                    #Hough 도메인의 값은 각 직선위의 엣지 픽셀의 개수를 의미 

    print("허프",Hough)

    for angle in range(0,180-1):
            for R in range(-Range+1, Range-1):
                isTrueLine = False
                if(Hough[angle][R] >= kThreshHoldLine): #누적 투표량이 임곗값 이상인 거리와 각도 
                    isTrueLine = True
                    print(Hough[angle][R])
                    
                    for dAngle in range(-1,1):
                        for dRho in range(-1,1):
                            if(Hough[angle+dAngle][R+dRho]>Hough[angle][R]):
                                isTrueLine=False
                            
                if(isTrueLine==True): #임곗값 이상의 점의 수로 구성된 직선 추출
                    angle_list.append(angle)
                    rho_list.append(R)

    img=np.zeros((n,m,3),np.uint8)
    image_arr=image_arr.astype(np.uint8)

    for i in range(len(angle_list)):
        print("angle",angle_list[i])
        a=np.cos(angle_list[i]*(np.pi / 180))
        b=np.sin(angle_list[i]*(np.pi / 180))
        x0=a*rho_list[i]
        y0=b*rho_list[i]

        scale=n+m

        x1=int(x0+scale*(-b))
        y1=int(y0+scale*a)
        x2=int(x0-scale*(-b))
        y2=int(y0-scale*a)

        
        img=cv2.line(image_arr,(x1,y1),(x2,y2),(0,255,0),2) #직선 표시 

    return img

def sobel(gaus_arr):
    dims=gaus_arr.shape
    n=dims[0]; m=dims[1]

    ix=np.copy(gaus_arr)
    iy=np.copy(gaus_arr)

    #소벨 연산 (1차 미분 )
    for i in range(1,n-1):
        for j in range(1,m-1):
            ix_=gaus_arr[i-1][j-1]*(-1)+gaus_arr[i][j-1]*0+gaus_arr[i+1][j-1]*1+gaus_arr[i-1][j]*(-2)+gaus_arr[i][j]*(0)+gaus_arr[i+1][j]*2+gaus_arr[i-1][j+1]*(-1)+gaus_arr[i][j+1]*0+gaus_arr[i+1][j+1]*1            
            iy_=gaus_arr[i-1][j-1]*(1)+gaus_arr[i][j-1]*2+gaus_arr[i+1][j-1]*1+gaus_arr[i-1][j]*(0)+gaus_arr[i][j]*(0)+gaus_arr[i+1][j]*0+gaus_arr[i-1][j+1]*(-1)+gaus_arr[i][j+1]*(-2)+gaus_arr[i+1][j+1]*(-1)
            
            ix[i][j]=ix_ #수직미분
            iy[i][j]=iy_ #수평미분 
    return (ix+iy)

#코너 검출
def corner(gaus_arr):

    dims=gaus_arr.shape
    n=dims[0]; m=dims[1]
    #ix, iy = np.gradient(gaus_arr) #1차미분계산 
    
    # 커널 형식 [0,1,0],[1,-4,1],[0,1,0]
    
    ix=np.copy(gaus_arr)
    iy=np.copy(gaus_arr)
    ix1=np.copy(gaus_arr)
    iy1=np.copy(gaus_arr)
    ix2=np.copy(gaus_arr)
    iy2=np.copy(gaus_arr)
    ixiy=np.copy(gaus_arr)

    print(ix1.shape, iy1.shape)
    for i in range(5):
        gaus_arr=Gaussian_filter(gaus_arr) #가우시안 필터 적용 

    #소벨 연산 (1차 미분 )
    for i in range(1,n-1):
        for j in range(1,m-1):
            ix_=gaus_arr[i-1][j-1]*(-1) + gaus_arr[i][j-1]*0 + gaus_arr[i+1][j-1]*1 + gaus_arr[i-1][j]*(-2)+gaus_arr[i][j]*(0)+gaus_arr[i+1][j]*2+gaus_arr[i-1][j+1]*(-1)+gaus_arr[i][j+1]*0+gaus_arr[i+1][j+1]*1            
            iy_=gaus_arr[i-1][j-1]*(1) + gaus_arr[i][j-1]*2+gaus_arr[i+1][j-1]*1+gaus_arr[i-1][j]*(0)+gaus_arr[i][j]*(0)+gaus_arr[i+1][j]*0+gaus_arr[i-1][j+1]*(-1)+gaus_arr[i][j+1]*(-2)+gaus_arr[i+1][j+1]*(-1)
            
            ix[i][j]=ix_ #수직미분
            iy[i][j]=iy_ #수평미분 
    
    # ix2=[0,]
    # iy2=[0,]
    # ixiy=[0,]

    # detM=np.copy(gaus_arr)
    # traceM=np.copy(gaus_arr)
    # R=np.copy(gaus_arr)

    ix=np.round_(ix,3)
    iy=np.round_(iy,3)

    print("ix iy",ix, iy)

    # return (ix+iy)
    # for i in range(1,n-1):
    #     for j in range(1,m-1):
    #         ix2[i][j]+=(ix[i][j] ** 2)
    #         iy2[i][j]+=(iy[i][j] ** 2)
    #         ixiy[i][j]+=(ix[i][j]*iy[i][j])
    # ix2=np.square(ix) #ix의 제곱값
    # iy2=np.square(iy)
    
    # ix2=ix**2
    # iy2=iy**2
    # ixiy=ix*iy
    # ix2=np.round_(ix2,3)
    # iy2=np.round_(iy2,3)
    # ixiy=np.round_(ixiy,3)

     #3*3 마스크 (sum)
    for i in range(1,n-1):
        for j in range(1,m-1):
            ix_=ix[i-1][j-1]**2+ix[i][j-1]**2+ix[i+1][j-1]**2+ix[i-1][j]**2+ix[i][j]**2+ix[i+1][j]**2+ix[i-1][j+1]**2+ix[i][j+1]**2+ix[i+1][j+1]**2            
            iy_=iy[i-1][j-1]**2+iy[i][j-1]**2+iy[i+1][j-1]**2+iy[i-1][j]**2+iy[i][j]**2+iy[i+1][j]**2+iy[i-1][j+1]**2+iy[i][j+1]**2+iy[i+1][j+1]**2
            ixiy_=ix[i-1][j-1]*iy[i-1][j-1]+ix[i][j-1]*iy[i][j-1]+iy[i+1][j-1]*ix[i+1][j-1]+iy[i-1][j]*ix[i-1][j]+iy[i][j]*ix[i][j]+iy[i+1][j]*ix[i+1][j]+iy[i-1][j+1]*ix[i-1][j+1]+iy[i][j+1]*ix[i][j+1]+iy[i+1][j+1]*ix[i+1][j+1]
            
            ix2[i][j]=ix_ #수직미분
            iy2[i][j]=iy_ #수평미분 
            ixiy[i][j]=ixiy_

    print("ix2",ix2)
    print("iy2", iy2)
    print("ixiy", ixiy)

    detM = (ix2 * iy2) - (ixiy ** 2) #det M
    traceM = (ix2 + iy2) #trance(M)
    print("detM",detM)
    #detM=np.round_(detM,3)
    # traceM=np.round_(traceM,3)

    # for i in range(1,n-1):
    #     for j in range(1,m-1):
    #         detM[i][j] += (ix2[i][j] * iy2[i][j]) - (ixiy[i][j] ** 2) #det M
    #         traceM[i][j] += (ix2[i][j] + iy2[i][j]) #trance(M)
    
    print("detM",detM)
    print("traceM",traceM)
    k=0.04 #k값은 보통 0.04로 함 

    R = detM - (k* (traceM ** 2)) #현재 윈도우의 R값 𝑅 = det 𝑀 − 𝑘(𝑡𝑟𝑎𝑐𝑒(𝑀))2
    # R=np.round_(R,3)

    corners = []
    print("R",R)

    for i in range(1, n-1):
        for j in range(1, m-1):
            # if R[i][j] >= max(R[i-1][j-1], R[i][j-1], R[i+1][j-1], R[i-1][j+1], R[i][j+1], R[i+1][j+1], R[i-1][j], R[i+1][j]): #센터값이 전체보다 더 클 경우  
            if(R[i][j]>0): #임곗값 
                corners.append((i, j, R[i][j])) #2개 고유값이이 둘다 클 경우, 코너점임
    
    print("corners",corners)
    # for i in range(1, R.shape[0] - 1):
    #     for j in range(1, R.shape[1] - 1):
    #         if R[i][j] >= max(R[i-1][j-1], R[i][j-1], R[i+1][j-1], R[i-1][j+1], R[i][j+1], R[i+1][j+1], R[i-1][j], R[i+1][j]): #센터값이 전체보다 더 클 경우 
    #             R[i][j]=round(R[i][j],5) #소수점 5째자리까지만 
    #             if(R[i][j]>(0)): #임곗값 
    #                 corners.append((i, j, R[i][j])) #2개 고유값이이 둘다 클 경우, 코너점임 
    
    return corners

    # dims=gaus_arr.shape
    # n=dims[0]; m=dims[1]
    # cor_arr=np.copy(gaus_arr)
    # corners_2=[]
    # for i in range(1,n-1):
    #     for j in range(1,m-1):
    #         lap=gaus_arr[i-1][j-1]+gaus_arr[i][j-1]+gaus_arr[i+1][j-1]+gaus_arr[i-1][j]+gaus_arr[i][j]*(-8)+gaus_arr[i+1][j]+gaus_arr[i-1][j+1]+gaus_arr[i][j+1]+gaus_arr[i+1][j+1]            
    #         if(lap>=max(gaus_arr[i-1][j-1], gaus_arr[i][j-1], gaus_arr[i+1][j-1], gaus_arr[i-1][j+1], gaus_arr[i][j+1], gaus_arr[i+1][j+1], gaus_arr[i-1][j], gaus_arr[i+1][j])):  
    #             corners_2.append((i,j))
    
    # corners_3=[]
    # dims=gaus_arr.shape
    # n=dims[0]; m=dims[1]
    # cor2_arr=np.copy(gaus_arr)
    # for i in range(n):
    #     for j in range(m):
    #         cor2_arr[i][j]=0
    
    # for i in range(1,n-1):
    #     for j in range(1,m-1):
            
    #         if(cor2_arr[i][j]>=max(gaus_arr[i-1][j-1], gaus_arr[i][j-1], gaus_arr[i+1][j-1], gaus_arr[i-1][j+1], gaus_arr[i][j+1], gaus_arr[i+1][j+1], gaus_arr[i-1][j], gaus_arr[i+1][j])):  
    #             corners_3.append((i,j))

def corner_image(image_arr, corners):
    x = [corner[0] for corner in corners]
    y = [corner[1] for corner in corners]

    for i in range(1,len(x)-1):
            image_arr[int(x[i])][int(y[i])]=(0,255,0)
    
    return image_arr

def mct(gray_arr): #3씩 증가하여 3*3 window로 평균값 구해 평균보다 크면 255, 작으면 0을 부여하고, 그 값을 배열에 저장하여 반환함 

    dims=gray_arr.shape
    n=dims[0]; m=dims[1]
    face_arr=np.copy(gray_arr)
    mean=0 # 평균(window 값 전체 합) 
    
    for i in range(n):
        for j in range(m):
            face_arr[i][j]=0
    
    for i in range(1,n-1,3): 
        for j in range(1,m-1,3):
            mean=gray_arr[i-1][j-1]+gray_arr[i][j-1]+gray_arr[i+1][j-1]+gray_arr[i-1][j]+gray_arr[i][j]+gray_arr[i+1][j]+gray_arr[i-1][j+1]+gray_arr[i][j+1]+gray_arr[i+1][j+1]
            mean = int(mean/9)
            # for di in range(-1,1,1):
            #     for dr in range(-1,1,1):
            #         if(gray_arr[i+di][j+dr]>mean): #평균보다 크면 1
            #             face_arr[i+di][j+dr]=255
            #         else:
            #             face_arr[i+di][j+dr]=0 #평균보다 작으면 0을 부여함

            if(gray_arr[i][j]>mean): #평균보다 크면 1
                face_arr[i][j]=255
            else: #평균보다 작으면 0을 부여함 
                face_arr[i][j]=0
            if(gray_arr[i-1][j-1]>mean): #평균보다 크면 1
                face_arr[i-1][j-1]=255
            else: #평균보다 작으면 0을 부여함 
                face_arr[i-1][j-1]=0
            if(gray_arr[i][j-1]>mean): #평균보다 크면 1
                face_arr[i][j-1]=255
            else: #평균보다 작으면 0을 부여함 
                face_arr[i][j-1]=0
            if(gray_arr[i+1][j-1]>mean): #평균보다 크면 1
                face_arr[i+1][j-1]=255
            else: #평균보다 작으면 0을 부여함 
                face_arr[i+1][j-1]=0
            if(gray_arr[i-1][j]>mean): #평균보다 크면 1
                face_arr[i-1][j]=255
            else: #평균보다 작으면 0을 부여함 
                face_arr[i-1][j]=0
            if(gray_arr[i+1][j]>mean): #평균보다 크면 1
                face_arr[i+1][j]=255
            else: #평균보다 작으면 0을 부여함 
                face_arr[i+1][j]=0
            if(gray_arr[i-1][j+1]>mean): #평균보다 크면 1
                face_arr[i-1][j+1]=255
            else: #평균보다 작으면 0을 부여함 
                face_arr[i-1][j+1]=0
            if(gray_arr[i][j+1]>mean): #평균보다 크면 1
                face_arr[i][j+1]=255
            else: #평균보다 작으면 0을 부여함 
                face_arr[i][j+1]=0
            if(gray_arr[i+1][j+1]>mean): #평균보다 크면 1
                face_arr[i+1][j+1]=255
            else: #평균보다 작으면 0을 부여함 
                face_arr[i+1][j+1]=0
    return face_arr

def downsampling(gray_arr): #다운 샘플링 (가우시안 피라미드) 짝수열,짝수행 픽셀 제거 1/2로 줄임 

    down_arr=gray_arr[::2,::2]
    print(down_arr.shape)
    return down_arr

def face_detection(gray_arr): #얼굴 검출 함수 3*3 윈도우로 얼굴인지 확인하기 위해 다운샘플링함  
    
    result_arr = [[0 for col in range(3)] for row in range(3)]
    face_arr=downsampling(gray_arr) #다운 샘플링함 (사이즈를 줄여서 다양한 얼굴도 인식가능하도록 함)

    for _ in range(2):
        face_arr=downsampling(face_arr) #다운 샘플링함 (사이즈를 줄여서 다양한 얼굴도 인식가능하도록 함)
        face_arr=mct(face_arr) #MCT 수행
        
        dims=face_arr.shape
        n=dims[0]; m=dims[1]
        
        for j in range(1,n-1): #높이 
            for h in range(1,m-1): #너비
                if((face_arr[j-1][h-1] == face_arr[j+1][h-1]) and (face_arr[j-1][h] == face_arr[j+1][h]) and (face_arr[j-1][h+1] == face_arr[j+1][h+1])):
                    result_arr[0][0]=face_arr[j-1][h-1]
                    result_arr[1][0]=face_arr[j][h-1]
                    result_arr[2][0]=face_arr[j+1][h-1]
                    result_arr[0][1]=face_arr[j-1][h]
                    result_arr[1][1]=face_arr[j][h]
                    result_arr[2][1]=face_arr[j+1][h]
                    result_arr[0][2]=face_arr[j-1][h+1]
                    result_arr[1][2]=face_arr[j][h+1]
                    result_arr[2][2]=face_arr[j+1][h+1]
    
    print(np.array(result_arr))

    return face_arr
#1. 엣지 검출

def EdgeDetection(image):
    image_arr = qimage2ndarray.rgb_view(image) #Qimage를 numpy로 변환
    gray_arr=Gray_scale(image_arr) #그레이 스케일 
    gray_arr=padding(gray_arr) #패딩
    gaus_arr=Gaussian_filter(gray_arr) #가우시안 필터
    for i in range(80):
        gaus_arr=Gaussian_filter(gaus_arr) #가우시안 필터
    
    lap_arr=Laplacian(gaus_arr) #라플라시안 필터
    print(lap_arr)
    zero_arr = zerocrossing(lap_arr)
    #print(zero_arr)
    image=qimage2ndarray.array2qimage(zero_arr, normalize=False) #numpy를 Qimage로 변환
    qPixmapVar = QPixmap.fromImage(image) #Qimage를 Qpixmap으로 변환   
    
    return qPixmapVar

#2. 직선 검출

def HoughTransform(image):
    image_arr=qimage2ndarray.rgb_view(image) #Qimage를 numpy로 변환
    gray_arr=Gray_scale(image_arr) #그레이 스케일
    gray_arr=padding(gray_arr) #패딩
    gaus_arr=Gaussian_filter(gray_arr) #가우시안 필터
    # for i in range(10):
    #     gaus_arr=Gaussian_filter(gaus_arr) #가우시안 필터
    lap_arr=Laplacian(gaus_arr) #라플라시안 필터 엣지검출함
    zero_arr = zerocrossing(lap_arr) 
    hou_arr=hough(image_arr, zero_arr)  #허프 변환
    image=qimage2ndarray.array2qimage(hou_arr, normalize=False) #numpy를 Qimage로 변환
    qPixmapVar = QPixmap.fromImage(image) #Qimage를 Qpixmap으로 변환  

    return qPixmapVar

#3. 코너 검출

def Harris_CornerDetection(image):
    image_arr=qimage2ndarray.rgb_view(image) #Qimage를 numpy로 변환
    gray_arr=Gray_scale(image_arr) #그레이 스케일
    gray_arr=padding(gray_arr) #패딩
    cor_arr=corner(gray_arr) #해리스 코너 검출 
    corner_result=corner_image(image_arr,cor_arr)
    image=qimage2ndarray.array2qimage(corner_result, normalize=False) #numpy를 Qimage로 변환
    qPixmapVar = QPixmap.fromImage(image) #Qimage를 Qpixmap으로 변환  

    return qPixmapVar

#4. 얼굴 검출 
def Face_Detection(image):
    image_arr=qimage2ndarray.rgb_view(image) #Qimage를 numpy로 변환
    gray_arr=Gray_scale(image_arr) #그레이 스케일
    gray_arr=padding(gray_arr) #패딩
    facedetection_arr=face_detection(gray_arr) #얼굴 검출 함수 
    image=qimage2ndarray.array2qimage(facedetection_arr, normalize=False) #numpy를 Qimage로 변환
    qPixmapVar = QPixmap.fromImage(image) #Qimage를 Qpixmap으로 변환  

    return qPixmapVar