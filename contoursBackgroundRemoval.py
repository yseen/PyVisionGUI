#Background removal, using edge detection and contours.
#Adapted from https://chris-s-park.medium.com/image-background-removal-using-opencv-part-1-da3695ac66b6
#Author: Yaseen Moolla
#Date: 2021-08-21

import numpy as np
import cv2
import time
import os
from glob import glob

'''
canny_low = 50
canny_high = 150
gauss_mask_size = 3

outDirname = 'bgRemoved'
if os.path.isdir(outDirname)==False:
    os.mkdir(outDirname)
inDirname = r'stable/*.jpeg'
'''

def findSignificantContour(edgeImg):
    contours, hierarchy = cv2.findContours(
        edgeImg,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
        # Find level 1 contours
    level1Meta = []
    for contourIndex, tupl in enumerate(hierarchy[0]):
        # Filter the ones without parent
        if tupl[3] == -1:
            tupl = np.insert(tupl.copy(), 0, [contourIndex])
            level1Meta.append(tupl)# From among them, find the contours with large surface area.
    contoursWithArea = []
    for tupl in level1Meta:
        contourIndex = tupl[0]
        contour = contours[contourIndex]
        area = cv2.contourArea(contour)
        contoursWithArea.append([contour, area, contourIndex])
        contoursWithArea.sort(key=lambda meta: meta[1], reverse=True)
    largestContour = contoursWithArea[0][0]
    return largestContour

'''
img_names = glob(inDirname)
img_names.sort(key=os.path.getmtime) #sort by date (because default sort by filename puts view10 before view2)
#read and process images.
imgList=[]
start = time.time()

for fn in img_names:
    print('processing %s...' % fn,)
    image_src = cv2.imread(fn)
'''
    
def removeBG(image_src, canny_low, canny_high, gauss_mask_size):     
    gauss_mask_size = int(gauss_mask_size)
    #print("gauss_mask_size ", gauss_mask_size)
    #gaussian blurring - gets rid of little details
    g_blurred = cv2.GaussianBlur(image_src, (gauss_mask_size, gauss_mask_size), 0)
    #blurred_float = g_blurred.astype(np.float32) / 255.0
    #image_src = g_blurred
    
    # Convert image to grayscale        
    image_gray = cv2.cvtColor(g_blurred, cv2.COLOR_BGR2GRAY)

    # Apply Canny Edge Dection
    #image_gray *= (255.0/image_gray.max())#convert range [0,1] to [0[255]
    image_gray = np.uint8(image_gray)
    edges = cv2.Canny(image_gray, canny_low, canny_high)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)
    edges_ = edges

    #determine contours
    contour = findSignificantContour(edges_)
    # Draw the contour on the original image
    contourImg = np.copy(image_src)
    cv2.drawContours(contourImg, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
    bgMask = np.zeros_like(edges_)
    cv2.fillPoly(bgMask, [contour], 255)# calculate sure foreground area by dilating the mask

    frame = image_src
    #combine mask with image
    # get first masked value (foreground)
    fg = cv2.bitwise_or(frame, frame, mask=bgMask)
    return fg
    
    
    '''
    cv2.imwrite('bitwiseOR.jpg', fg)
    # get second masked value (background) mask must be inverted
    altMask = cv2.bitwise_not(bgMask)
    background = np.full(frame.shape, 255, dtype=np.uint8)
    bk = cv2.bitwise_or(background, background, mask=altMask)
    # combine foreground+background
    final = cv2.bitwise_or(fg, bk)
    '''

'''
    outfile = os.path.join(outDirname,os.path.basename(fn))
    print("writing to: ", outfile)
    
    cv2.imwrite(outfile, fg)
end = time.time()
totalTime = end - start                
print("processing time: ", totalTime)
'''