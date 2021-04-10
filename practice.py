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
    global img1,tempimg
    if (counter % 2 == 0 and counter > 0):
        if event == cv2.EVENT_LBUTTONDOWN:
            px1=x
            py1=y
        elif event ==cv2.EVENT_LBUTTONUP:    #滑鼠鬆開停止畫線
            qx1=x
            qy1=y
            cv2.line(img1, (int(px1), int(py1)), (int(qx1), int(qy1)), (0, 0, 255), 3)
            tempimg=img1.copy()
            cv2.imshow('left',img1)
            tleft = [px1,py1,qx1,qy1]  #左圖線段
            lline.append(tleft) #記錄左圖線段 
            counter=counter-1
        elif event == cv2.EVENT_FLAG_LBUTTON:
            tempimg=img1.copy()
            cv2.imshow('left',img1)
#右視窗中畫feature line
def draw2(event, x, y, flags, param):
    global tleft ,tright,lline,rline
    global counter
    global px2,py2,qx2,qy2
    global img2,tempimg
    if (counter % 2 == 1 and counter > 0):
        if event == cv2.EVENT_LBUTTONDOWN:
            px2=x
            py2=y
        elif event ==cv2.EVENT_LBUTTONUP:    #滑鼠鬆開停止畫線
            qx2=x
            qy2=y
            cv2.line(img2, (int(px2), int(py2)), (int(qx2), int(qy2)), (0, 0, 255), 3)
            tempimg=img2.copy()
            cv2.imshow('right',img2)
            tright = [px2,py2,qx2,qy2] #右圖線段 
            rline.append(tright) #記錄右圖線段 
            counter=counter-1
        elif event == cv2.EVENT_FLAG_LBUTTON:
            tempimg=img2.copy()
            cv2.imshow('right',img2)

# def showWarpLine():
    
#     for i in range(len(warpLine)):
#         px,py,qx,qy=warpLine[0],line[1],line[2],line[3]
#         cv2.line(img1, (int(px2), int(py2)), (int(qx2), int(qy2)), (0, 0, 255), 3)

# 	return

def pqtomld(line): #已知PQ線段 算出中點,長度,角度
    px,py,qx,qy=line[0],line[1],line[2],line[3]
    mx=(px+qx)/2 #中點
    my=(py+qy)/2 
    length = math.sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py)) #長度
    degree = math.atan2((qy-py),(qx-px)) #角度
    mld=[mx,my,length,degree]
    return mld

#mld=[mx,my,length,degree]
def mldtopq(mldlist): #已知中點,長度,角度 算出PQ線段
    mx,my,length,degree=mldlist[0],mldlist[1],mldlist[2],mldlist[3]
    tmpx = 0.5 * length * math.cos(degree)
    tmpy = 0.5 * length * math.sin(degree)
    px = mx - tmpx
    py = my - tmpy
    qx = mx + tmpx
    qy = my + tmpy
    line = [px,py,qx,qy]
    return line

# show & draw images     
cv2.namedWindow('left')
img1 = cv2.imread("image/women.jpg", cv2.IMREAD_COLOR)
cv2.setMouseCallback('left', draw1,img1)
cv2.namedWindow('right')
img2 = cv2.imread("image/cheetah.jpg", cv2.IMREAD_COLOR)
cv2.setMouseCallback('right', draw2,img2)
rows,cols = img1.shape[0],img1.shape[1]
new_image = np.zeros((rows,cols,3), np.uint8)
left_image = np.zeros((rows,cols,3), np.uint8)
right_image = np.zeros((rows,cols,3), np.uint8)
tempimg=np.zeros((rows,cols,3), np.uint8)

# def changepixel(): 
#     cv2.namedWindow('test')
#     imgtest = img1.copy()
#     rows,cols = imgtest.shape[0],imgtest.shape[1]
#     for i in range(rows): #rows
#         for j in range(cols): #cols
#             imgtest[i,j]=(255,0,0) #OpenCV中圖片BGR
#     cv2.imshow('test', imgtest)

# def writeimg():
#     cv2.namedWindow('test')
#     rows,cols = img1.shape[0],img1.shape[1]
#     blankimg = np.zeros((rows,cols,3), np.uint8)
#     for i in range(rows): #rows
#         for j in range(cols): #cols
#             color=(img1[i,j]+img2[i,j])/2
#             #blankimg[i,j]=color #OpenCV中圖片BGR
#     cv2.imshow('test', blankimg)
#     cv2.imwrite('write.jpg', blankimg)

def getu(x,y,line): #算出該點在線段獲得的u
    px,py,qx,qy=line[0],line[1],line[2],line[3]
    X_Px = x-px
    X_Py = y-py
    Q_Px = qx - px
    Q_Py = qy - py
    length = math.sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py)) #長度
    u = ((X_Px * Q_Px) + (X_Py * Q_Py)) / (length * length)
    return u

def getv(x,y,line):#算出該點在線段獲得的v
    px,py,qx,qy=line[0],line[1],line[2],line[3]
    X_Px = x - px
    X_Py = y - py
    Q_Px = qx - px
    Q_Py = qy - py
    Perp_QPx = Q_Py
    Perp_QPy = -Q_Px
    length = math.sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py)) #長度
    v = ((X_Px * Perp_QPx) + (X_Py * Perp_QPy)) / length
    return v

def getweight(x,y,line):
    px,py,qx,qy=line[0],line[1],line[2],line[3]
    a,b,p=1,2,2
    d = 0.0
    u = getu(x,y,line)
    if (u > 1.0):
        d = math.sqrt((x - qx) * (x - qx) + (y - qy) * (y - qy)) 
    elif (u < 0):
        d = math.sqrt((x - px) * (x - px) + (y - py) * (y - py))
    else:
        d = abs(getv(x,y,line))
    length = math.sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py)) #長度
    weight = math.pow(math.pow(length, p) / (a + d), b)
    return weight

def getpoint(u,v,line):
    px,py,qx,qy=line[0],line[1],line[2],line[3]
    qpx=qx-px
    qpy=qy-py
    perqpx=qpy
    perqpy=-qpx
    length = math.sqrt((qx-px)*(qx-px)+(qy-py)*(qy-py)) #長度
    pointx=px+u*(qx-px)+((v*perqpx)/length)
    pointy=py+u*(qy-py)+((v*perqpy)/length)
    point=[pointx,pointy]
    return point

def caldegree(line):
    px,py,qx,qy=line[0],line[1],line[2],line[3]
    degree = math.atan2((qy - py), (qx - px))
    return degree

def bilinear(img,x,y): #(image,srcx,srcy)
    width,height=img.shape[0],img.shape[1]
    x_floor = int(x)
    y_floor = int(y)
    x_ceil = int(x)+1
    y_ceil = int(y)+1
    a,b=x-x_floor,y-y_floor
    if (x_ceil >= width - 1):
        x_ceil = width - 1
    if (y_ceil >= height - 1):
        y_ceil = height - 1

    leftdown=img[x_floor,y_floor]
    lefttop=img[x_floor,y_ceil]
    rightdown=img[x_ceil,y_floor]
    righttop=img[x_ceil,y_ceil]  
    
    # leftdown=img[y_floor, x_floor]
    # lefttop=img[y_ceil, x_floor]
    # rightdown=img[y_floor, x_ceil]
    # righttop=img[y_ceil, x_ceil]

    out=[0,0,0]
    for i in range (3):
        out[i]=(1 - a) * (1 - b) * leftdown[i] + a * (1 - b) * rightdown[i] + a * b * righttop[i] + (1 - a) * b * lefttop[i]
    
    return out

warpLine=[]
frame_count=1
def genWarpLine():
    global frame_count
    global warpLine
    warpLine.clear()
    for i in range (len(lline)):
        ldegree,rdegree=caldegree(lline[i]),caldegree(rline[i])
        while(ldegree-rdegree>math.pi):
            rdegree=rdegree+math.pi
        while(rdegree-ldegree>math.pi):
            ldegree=ldegree+math.pi
        for j in range (frame_count): #frame count
            ratio=(j+1)/(frame_count+1)
            mldleft=pqtomld(lline[i])
            mldright=pqtomld(rline[i])
            mx = (1 - ratio) * mldleft[0] + ratio * mldright[0]
            my = (1 - ratio) * mldleft[1] + ratio * mldright[1]
            length = (1 - ratio) * mldleft[2] + ratio * mldright[2]
            degree = (1 - ratio) * mldleft[3] + ratio * mldright[3]
            mldlist=[mx,my,length,degree]
            pqline=mldtopq(mldlist)
            warpLine.append(pqline)
    print(len(warpLine))
    print(warpLine)
    return


###############################
def runwrap():
    print('in runwrap')
    genWarpLine()
    wrapping()
    pic1 = cv2.imread("result/left.jpg")
    pic2 = cv2.imread("result/right.jpg")
    dst=cv2.addWeighted(pic1, 0.5, pic2, 0.5, 0.0)
    cv2.imwrite("result/half.jpg", dst)

def wrapping():
    print('in wrapping')
    #original img: img1 & img2
    img1 = cv2.imread("image/women.jpg", cv2.IMREAD_COLOR)
    img2 = cv2.imread("image/cheetah.jpg", cv2.IMREAD_COLOR)
    ratio = 0.5
    #ratio = (frame_index + 1) / (frame_count + 1)
    for x in range(rows): #rows
        for y in range(cols): #cols
            dstx=x
            dsty=y
            leftxsum,leftysum,leftWeightSum,rightxsum,rightysum,rightWeightSum=0.0,0.0,0.0,0.0,0.0,0.0
            for i in range(len(lline)):
                #左圖為來源
                #leftline = lline[i] = (px1 py1 qx1 qy1) 
                dstline=[] #(dpx dpy dqx dqy)
                src_line = lline[i]
                dst_line = warpLine[frame_count*i] #frame_count*i
                newu = getu(dstx,dsty,dst_line) #(x,y,line)
                newv = getv(dstx,dsty,dst_line) #(x,y,line)
                src_point = getpoint(newu, newv,src_line) #(u,v,line)
                srcx,srcy = src_point[0],src_point[1]
                src_weight = getweight(dstx,dsty,dst_line) #(x,y,line)
                leftxsum=leftxsum+srcx*src_weight
                leftysum=leftysum+srcy*src_weight
                leftWeightSum=leftWeightSum+src_weight

                #右圖為來源 
                #rightline = rline[i] = (px2 py2 qx2 qy2)
                dstline=[] #(dpx dpy dqx dqy)
                src_line = rline[i]
                dst_line = warpLine[frame_count*i]
                newu = getu(dstx,dsty,dst_line) #(x,y,line)
                newv = getv(dstx,dsty,dst_line) #(x,y,line)
                src_point = getpoint(newu, newv,src_line) #(u,v,line)
                srcx,srcy = src_point[0],src_point[1]
                src_weight = getweight(dstx,dsty,dst_line) #(x,y,line)
                rightxsum = rightxsum+srcx*src_weight
                rightysum = rightysum+srcy*src_weight
                rightWeightSum=rightWeightSum+src_weight

            lsrcx=leftxsum/leftWeightSum
            lsrcy=leftysum/leftWeightSum
            rsrcx=rightxsum/rightWeightSum
            rsrcy=rightysum/rightWeightSum

            #邊界
            if (lsrcx < 0):
                lsrcx = 0
            if (lsrcy < 0):
                lsrcy = 0
            if (lsrcx >= rows):
                lsrcx = rows - 1
            if (lsrcy >= cols):
                lsrcy = cols - 1
            if (rsrcx < 0):
                rsrcx = 0
            if (rsrcy < 0):
                rsrcy = 0
            if (rsrcx >= rows):
                rsrcx = rows - 1
            if (rsrcy >= cols):
                rsrcy = cols - 1

            leftout=bilinear(img1,lsrcx,lsrcy)
            rightout=bilinear(img2,rsrcx,rsrcy)
            
            b=(1 - ratio)*leftout[0]+ratio*rightout[0]
            g=(1 - ratio)*leftout[1]+ratio*rightout[1]
            r=(1 - ratio)*leftout[2]+ratio*rightout[2]

            new_image[x,y]=(b,g,r)
            left_image[x,y]=(b,g,r)
            right_image[x,y]=(b,g,r)

    cv2.imwrite("result/new.jpg", new_image)
    cv2.imwrite("result/left.jpg", left_image)
    cv2.imwrite("result/right.jpg", right_image)

        #######################################


#add weight
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
        # changepixel()
        #writeimg()
        # pqtomld(rline[0])
        # addimg()
        # wrapping()
        # print(getu(3,10,rline[0]))
        # print(getv(3,5,rline[0]))
        #print(getweight(3,5,rline[0]))
        # genWarpLine()
        runwrap()

cv2.destroyAllWindows()