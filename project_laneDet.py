# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

#%matplotlib inline

import math

def imageDims(image):
    imshape = image.shape
    x= imshape[0]
    y= imshape[1]
    return x,y

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_hsv(img):
    """convert BGR to HSV"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def gradient(line):
    """ calculates gradient/slope of a line """
    for x1,y1,x2,y2 in line:
        m = (y2*1.00-y1)/(x2*1.00-x1)
        return m
    
def y_int(line,m):
    """ calculates y-intercept of given line """
    for x1,y1,x2,y2 in line:
        c = y1 - (m*x1)     # y = mx+c  =>  c = y - mx
        return c
    
def generate_points(mean_intercept,mean_slope,x):
    """
    generates to sets of x,y coordinates for averaged/extrapolated line.
    With the given slope and y-intercept, uses the equation:
        y = mx + c
    using x values from bottom of image to top of ROI
    """
    y1 = x*0.6  # roi top
    x1 = (y1 - mean_intercept)/mean_slope
    
    y2 = x      # roi bottom x
    x2 = (y2 - mean_intercept)/mean_slope

    x1 = int(round(x1))
    x2 = int(round(x2))
    y1 = int(round(y1))
    y2 = int(round(y2))

    return x1,y1,x2,y2


def extrapolate(slope,inter,PREV_SLOPE_MEAN,PREV_INT_MEAN,window,index):
    slope_mean = np.mean(slope)
    inter_mean = np.mean(inter)

    # if averaging window not full, append mean values
    # otherwise replace the oldest value
    if (len(PREV_SLOPE_MEAN) != window): 
        PREV_SLOPE_MEAN.append(slope_mean)
        PREV_INT_MEAN.append(inter_mean)
    else:
        PREV_SLOPE_MEAN[index] = slope_mean
        PREV_INT_MEAN[index] = inter_mean

    # calculate new slope and intercept using moving average window
    new_slope = np.mean(PREV_SLOPE_MEAN)
    new_inter = np.mean(PREV_INT_MEAN)

    return new_slope,new_inter


PREV_L_SLOPE_MEAN = []
PREV_R_SLOPE_MEAN = []
PREV_L_INT_MEAN = []
PREV_R_INT_MEAN = []
COUNT_R = 0
COUNT_L = 0


def draw_lines(img, lines, color=[0, 0, 255], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """

    global PREV_L_SLOPE_MEAN ,PREV_R_SLOPE_MEAN
    global PREV_L_INT_MEAN, PREV_R_INT_MEAN
    global COUNT_L, COUNT_R
    
    x,y = imageDims(img)
    left_bound = y*.45  #bounds for determining left or right lane 
    right_bound = y*.55

    l_slope = []    # slopes of lines for current frame (left lane)
    r_slope = []    # slopes of lines for current frame (right lane)
    
    l_inter = []    # y-intercepts of lines for current frame (left lane)
    r_inter = []    # y-intercepts of lines for current frame (right lane)

    window = 20     # range of moving average window for feedback
    r_index = COUNT_R % window  # index to update oldest value
    l_index = COUNT_L % window  # index to update oldest value
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            
            m = gradient(line)
            if (m>-0.2 and m<0.2):  # disregard horizonal lines
                continue
            c = y_int(line, m)

            # separate right and left lane lines 
            if ( (right_bound - x1) < (x1- left_bound) ) :
                r_slope.append(m)
                r_inter.append(c)
            else:
                l_slope.append(m)
                l_inter.append(c)
                
    if (len(l_slope)!=0 ): 
        slope,inter = extrapolate(l_slope,l_inter,PREV_L_SLOPE_MEAN,PREV_L_INT_MEAN,window,l_index)
        x1,y1,x2,y2 = generate_points(inter,slope,x)
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        COUNT_L +=1
        COUNT_L = COUNT_L%100
        
    else:
        # if no lines detected, use previous values
        if(len(PREV_L_SLOPE_MEAN)!=0):
            slope = np.mean(PREV_L_SLOPE_MEAN)
            inter = np.mean(PREV_L_INT_MEAN)
            x1,y1,x2,y2 = generate_points(inter,slope,x)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        # do same for right lane lines
    if (len(r_slope)!=0 ):

        slope,inter = extrapolate(r_slope,r_inter,PREV_R_SLOPE_MEAN,PREV_R_INT_MEAN,window,r_index)
        x1,y1,x2,y2 = generate_points(inter,slope,x) 
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        COUNT_R +=1
        COUNT_R = COUNT_R%100  # reset COUNT every 100
    else:
        if(len(PREV_R_SLOPE_MEAN)!=0):
            slope = np.mean(PREV_R_SLOPE_MEAN)
            inter = np.mean(PREV_R_INT_MEAN)
            x1,y1,x2,y2 = generate_points(inter,slope,x)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def color_mask(img,hsv):
    """
    color ranges for white and yellow hsv to create mask
    combine yellow and white masks
    apply mask to image and returns the masked image
    """
    lower_white = np.array([0/2,0,191]) #0,0,75%
    upper_white = np.array([359/2,25,255])#359,10%,100%
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lower_yellow = np.array([36/2,63,178]) #36,25%,70%
    upper_yellow = np.array([60/2,255,255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(mask_white,mask_yellow)
    maskedImage = cv2.bitwise_and(img,img, mask= mask)
    return maskedImage

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, a=0.8, b=1., l=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, a, img, b, l)



def process_image(image):
    # TODO: Build your pipeline that will draw lane lines on the test_images
    # then save them to the test_images directory.
    img = np.copy(image)
    x,y = imageDims(img)

    kernel_size = 5
    blurred = gaussian_blur(img, kernel_size)
    hsv = to_hsv(blurred)

    maskedImage = color_mask(img,hsv)

    vertices = np.array([[(0,x),(y*.45, x*.6), (y*.55, x*.6), (y,x)]], dtype=np.int32)
    roi_img= region_of_interest(maskedImage, vertices)

    edges= canny(roi_img, low_threshold=100, high_threshold=300)
    rho = 1
    theta = np.pi/180
    threshold = 15 #15
    min_line_len = 10  #2
    max_line_gap = 20 #20
    lines= hough_lines(edges, rho, theta, threshold, min_line_len, max_line_gap)

    result= weighted_img(lines, img, a=0.8, b=1., l=0.)

    return result,lines,edges




#reading in an image

import os
import time
listOfTestImgs = os.listdir("test_images/")
cap = cv2.VideoCapture('test_videos/challenge.mp4');

fname = 'test_images/'+listOfTestImgs[4]
image = cv2.imread(fname)
res,lines,edges = process_image(image)
cv2.imshow('image2',res)
while(1):
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
'''
while (cap.isOpened()):
    time.sleep(0.0)
    ret, frame = cap.read();
    res,lines,edges = process_image(frame)
    cv2.imshow('image1',res)
    #cv2.imshow('image2',lines)
    #cv2.imshow('image3',edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

'''
