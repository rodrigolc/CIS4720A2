#/bin/env python3

import cv2
import cv2.cv as cv
import numpy as np
import sys

if not len(sys.argv) == 2:
    print("Wrong number of arguments")
    sys.exit(1)

filename = sys.argv[1]

image = cv2.imread(filename)


def dist_line(p1,p2,p3):
    x1,y1 = p1
    x2,y2 = p2
    x0,y0 = p3
    return ((y2 - y1) * x0 - (x2 - x1) * y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2 - x1) ** 2)

def dist_point(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return np.sqrt((y2-y1)**2 + (x2 - x1) ** 2)

def slope(x0, y0, x1, y1):
    if x0 == x1:
        return float("inf") if y0 < y1 else -float("inf")
    return (y1-y0)/float(x1-x0)


def fullLine(img, a, b, color):
    slope_ = slope(a[0], a[1], b[0], b[1])

    if slope_ == float("inf"):
        p = (a[0],0)
        q = (a[0],height)
    else:
        p,q = (0,0), (width,height)
        p = (p[0],int(-(a[0] - p[0]) * slope_ + a[1]))
        q = (q[0],int(-(b[0] - q[0]) * slope_ + b[1]))

    cv2.line(img,p,q,color,2)

def gauss_blur(image,x,y,distance):
    width = len(image[0])
    height = len(image)
    radius = int(np.ceil(distance))
    kernel = cv2.getGaussianKernel( 2*radius+1,distance)
    kernel = kernel * np.transpose(kernel)

    x_lcut = 0
    x_rcut = len(kernel[0])
    y_tcut = 0
    y_bcut = len(kernel)
    min_x = max(x - radius, 0)
    max_x = min(x + radius, len(image[0])-1)
    min_y = max(y - radius, 0)
    max_y = min(y + radius, len(image)-1)
    roi = image[min_y:max_y+1, min_x:max_x+1]
    y_offset = y - radius
    x_offset = x - radius

    mask_ = kernel[min_y - y_offset:max_y-y_offset + 1, min_x - x_offset : max_x - x_offset + 1]

    return np.average(roi,None,mask_)

def saturate(image,value):
    im2 = cv2.cvtColor(image,cv.CV_BGR2HSV)

    for i in range(len(image)):
        for j in range(len(image[0])):
            if im2[i][j][1] + value > 255:
                im2[i][j][1] = 255
            else:
                im2[i][j][1] += value

    return cv2.cvtColor(im2, cv.CV_HSV2BGR)


def tilt_shift(image,p1,p2,type,distance,ratio):
    nimg = np.empty_like(image)
    imageB = image[:,:,0]
    imageG = image[:,:,1]
    imageR = image[:,:,2]
    for i in range(len(image)):
        if mode == 0: #tilt-shift
            for j in range(len(image[i])):
                dist = dist_line(p1,p2,(j,i))
                if dist < 0:
                    dist = - dist
                if dist > distance:
                    nimg[i][j][0] = gauss_blur(imageB,j,i, (dist - distance) * ratio) #B
                    nimg[i][j][1] = gauss_blur(imageG,j,i, (dist - distance) * ratio) #G
                    nimg[i][j][2] = gauss_blur(imageR,j,i, (dist - distance) * ratio) #G
                else:
                    nimg[i][j] = image[i][j]
        elif mode == 1: #circle
            for j in range(len(image[i])):
                dist = dist_point(p1,(j,i))
                if dist < 0:
                    dist = - dist
                if dist > distance:
                    nimg[i][j][0] = gauss_blur(imageB,j,i, (dist - distance) * ratio) #B
                    nimg[i][j][1] = gauss_blur(imageG,j,i, (dist - distance) * ratio) #G
                    nimg[i][j][2] = gauss_blur(imageR,j,i, (dist - distance) * ratio) #R
                else:
                    nimg[i][j] = image[i][j]
    return nimg

height = len(image)
width = len(image[0])

ix = 0
iy = height/2
ix2 = width - 1
iy2 = height/2


mode = 0 #tilt_shift = 0, circle = 1

# mouse callback function
def get_mouse_point(event,x,y,flags,param):
    global ix,iy,ix2,iy2,image,width,height,mode
    if event == 1: #press down left button
        ix,iy = x,y
    if event == 4: #release left button
        ix2,iy2 = x,y
        aux = np.copy(image)
        cv2.line(aux,(ix,iy),(ix2,iy2),(200,100,50),2)
        if mode == 0: #tilt shift
            dx = ix2 - ix
            dy = iy2 - iy

            #same line, offset by distance between (ix,iy) and (ix2,iy2)
            y1 = iy - dx
            x1 = ix + dy
            y2 = iy2 - dx
            x2 = ix2 + dy
            fullLine(aux,(x1,y1),(x2,y2),(100,200,50))

            # same, other direction
            y1 = iy + dx
            x1 = ix - dy
            y2 = iy2 + dx
            x2 = ix2 - dy
            fullLine(aux,(x1,y1),(x2,y2),(100,200,50))

            cv2.imshow("image",aux)

        elif mode == 1: #circle
            radius = dist_point((ix,iy),(ix2,iy2))
            cv2.circle(aux,(ix,iy),int(radius),(100,200,50),2)
            cv2.imshow("image",aux)


cv2.namedWindow('image')
cv2.setMouseCallback('image',get_mouse_point)
cv2.imshow("image",image)

while True:
    key = cv2.waitKey(0)
    if key == 13: #enter
        break
    elif key == 116: # t = toggle modes
        mode = 1 - mode
        cv2.imshow("image",image)


dist = dist_point((ix,iy),(ix2,iy2))

nimg = tilt_shift(image,(ix,iy),(ix2,iy2),mode,dist,0.03)

simg = saturate(nimg,40)

cv2.imshow("blurred",nimg)
cv2.imshow("blurred and saturated",simg)

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("tilt_shift_" + filename.split('/')[-1],simg)
