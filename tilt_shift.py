#/bin/env python3

import cv2
import cv2.cv as cv
import numpy as np
import sys
from scipy.ndimage.filters import generic_filter as gf

if not len(sys.argv) == 2:
    print("Wrong number of arguments")
    sys.exit(1)

filename = sys.argv[1]

image = cv2.imread(filename)

# cv2.namedWindow('original image')
# cv2.setMouseCallback('original image',get_point)

def circular_blur(image,x,y,radius_):
    width = len(image[0])
    height = len(image)
    radius = int(np.ceil(radius_))
    kernel = cv2.getGaussianKernel( 2*radius+1,radius_)
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

def dist_line(p1,p2,p3):
    x1,y1 = p1
    x2,y2 = p2
    x0,y0 = p3
    return ((y2 - y1) * x0 - (x2 - x1) * y0 + x2*y1 - y2*x1) / np.sqrt((y2-y1)**2 + (x2 - x1) ** 2)

def dist_point(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    return np.sqrt((y2-y1)**2 + (x2 - x1) ** 2)

def gaussian_blur(image,x,y,distance):
    kernel = cv2.getGaussianKernel(25, distance)


def square_blur(image,x,y,distance):
    min_x = max(x - distance, 0)
    max_x = min(x + distance, len(image[0]))
    min_y = max(y - distance, 0)
    max_y = min(y + distance, len(image))
    roi = image[min_y:max_y+1, min_x:max_x+1]
    roi = cv2.resize(roi,(1,1),interpolation=cv.CV_INTER_LINEAR)
    return roi[0][0]

def tilt_shift(image,p1,p2,type,distance,ratio):
    nimg = np.empty_like(image)
    for i in range(len(image)):
        print i
        print distance
        for j in range(len(image[i])):
            dist = dist_line(p1,p2,(j,i))
            if dist < 0:
                dist = - dist
            if dist > distance:
                nimg[i][j][0] = circular_blur(image[:,:,0],j,i, (dist - distance) * ratio)
                nimg[i][j][1] = circular_blur(image[:,:,1],j,i, (dist - distance) * ratio)
                nimg[i][j][2] = circular_blur(image[:,:,2],j,i, (dist - distance) * ratio)
            else:
                nimg[i][j] = image[i][j]
    return nimg

height = len(image)
width = len(image[0])

ix,iy = -1,-1
ix2,iy2 = -1,-1
# mouse callback function
def get_mouse_point(event,x,y,flags,param):
    global ix,iy,ix2,iy2
    print "move", x,y
    print event
    if event == 1: #press down left button
        ix,iy = x,y
    if event == 4: #release left button
        ix2,iy2 = x,y

if ix == -1 and iy == -1:
    ix = 0
    iy = height/2

if ix2 == -1 and iy2 == -1:
    ix2 = width - 1
    iy2 = height/2


cv2.namedWindow('image')
cv2.setMouseCallback('image',get_mouse_point)
cv2.imshow("image",image)
cv2.waitKey(0)
print (ix,iy)
print (ix2,iy2)
dist = dist_point((ix,iy),(ix2,iy2))


def saturate(image,value):
    im2 = cv2.cvtColor(image,cv.CV_BGR2HSV)

    for i in range(len(image)):
        for j in range(len(image[0])):
            if im2[i][j][1] + value > 255:
                im2[i][j][1] = 255
            else:
                im2[i][j][1] += value

    return cv2.cvtColor(im2, cv.CV_HSV2BGR);


nimg = tilt_shift(image,(ix,iy),(ix2,iy2),0,dist/2,0.05)

simg = saturate(nimg,30)

cv2.imshow("hai",nimg)
cv2.imshow("hais",simg)
