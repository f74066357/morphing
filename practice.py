import cv2
import glob
import os
import math
from PIL import Image
import numpy as np
counter=0
px1=py1=qx1=qy1=0
px2=py2=qx2=qy2=0
# P 起點 Q 終點 
#左圖 tleft = (px1 py1 qx1 qy1) 
#右圖 tright = (px2 py2 qx2 qy2)
tleft =[]
tright =[]
lline = []
rline = []

#左視窗中畫feature line
def draw1(event, x, y, flags, param):
    global tleft ,tright,lline,rline
    global counter
    global px1,py1,qx1,qy1
    if (counter % 2 == 0 and counter > 0):
        if event == cv2.EVENT_LBUTTONDOWN:
            px1=x
            py1=y
        elif event ==cv2.EVENT_LBUTTONUP:    #滑鼠鬆開停止畫線
            qx1=x
            qy1=y
            cv2.line(param, (int(px1), int(py1)), (int(qx1), int(qy1)), (0, 0, 255), 3)
            tleft = [px1,py1,qx1,qy1]  #左圖線段
            lline.append(tleft) #記錄左圖線段 
            counter=counter-1
#右視窗中畫feature line
def draw2(event, x, y, flags, param):
    global tleft ,tright,lline,rline
    global counter
    global px2,py2,qx2,qy2
    if (counter % 2 == 1 and counter > 0):
        if event == cv2.EVENT_LBUTTONDOWN:
            px2=x
            py2=y
        elif event ==cv2.EVENT_LBUTTONUP:    #当鼠标松开时停止绘图
            qx2=x
            qy2=y
            cv2.line(param, (int(px2), int(py2)), (int(qx2), int(qy2)), (0, 0, 255), 3)
            tright = [px2,py2,qx2,qy2] #右圖線段 
            rline.append(tright) #記錄右圖線段 
            counter=counter-1

def pqtomld(line): #已知PQ點 算出中點,長度,角度
    #print(line)
    px=line[0]
    py=line[1]
    qx=line[2]
    qy=line[3]
    middle=[(px+qx)/2,(py+qy)/2] #中點
    length = math.sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py)) #長度
    degree = math.atan2((qy-py),(qx-px)) #角度
    #print(middle,length,degree)



# show & draw images     
cv2.namedWindow('left')
img1 = cv2.imread("image/women.jpg", cv2.IMREAD_COLOR)
cv2.setMouseCallback('left', draw1,img1)
cv2.namedWindow('right')
img2 = cv2.imread("image/cheetah.jpg", cv2.IMREAD_COLOR)
cv2.setMouseCallback('right', draw2,img2)

# def changepixel(): 
#     cv2.namedWindow('test')
#     imgtest = img1.copy()
#     rows,cols = imgtest.shape[0],imgtest.shape[1]
#     for i in range(rows): #rows
#         for j in range(cols): #cols
#             imgtest[i,j]=(255,0,0) #OpenCV中圖片BGR
#     cv2.imshow('test', imgtest)

def writeimg():
    cv2.namedWindow('test')
    rows,cols = img1.shape[0],img1.shape[1]
    blankimg = np.zeros((rows,cols,3), np.uint8)
    for i in range(rows): #rows
        for j in range(cols): #cols
            color=(img1[i,j]+img2[i,j])/2
            #blankimg[i,j]=color #OpenCV中圖片BGR
    cv2.imshow('test', blankimg)

def wrapping():
    #original img: img1 & img2
    ratio=0.5
    for x in range(rows): #rows
        for y in range(cols): #cols
            dstpoint_x=x
            dstpoint_y=y
            leftXSum_x = 0.0
            leftXSum_y = 0.0
            leftWeightSum = 0.0
            rightXSum_x = 0.0
            rightXSum_y = 0.0
            rightWeightSum = 0.0
            for (i in range (len(lline))):
                #左圖為來源
                #leftline = lline[i] = (px1 py1 qx1 qy1) 
                dstline=[] #(dpx dpy dqx dqy)
                
                #右圖為來源 
                #rightline = rline[i] = (px2 py2 qx2 qy2)


alpha=0.5
def addimg():
    global alpha
    cv2.namedWindow('add')
    alpha=alpha+0.1
    beta=1-alpha
    img= cv2.addWeighted(img1, alpha, img2, beta, 0.0)
    cv2.imshow('add', img)

while (1):
    cv2.imshow('left', img1)
    cv2.imshow('right', img2)
    if cv2.waitKey(1)&0xFF == ord('q'):#按q退出
        print('exit!')
        break
    elif cv2.waitKey(1)&0xFF == ord('s'): #按s顯示feature line
        print('feature line pair num:'+str(len(lline)))
        print('left image feature line:')
        print(lline)
        print('right image feature line:')
        print(rline)
    elif cv2.waitKey(1)&0xFF == ord('c'): #按c繪製feature line
        counter = counter + 2
        print("draw feature line (left&right)!")
    elif cv2.waitKey(1)&0xFF == ord('w'): #按w wrapping
        print("start wrapping")
    elif cv2.waitKey(1)&0xFF == ord('t'): #按t test
        #changepixel()
        #writeimg()
        #pqtomld(rline[0])
        #addimg()
        wrapping()

cv2.destroyAllWindows()