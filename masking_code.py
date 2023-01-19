# Nikhil Uday Shinde: 7/23/18
# https://github.com/nikhilushinde/cs194-26_proj3_2.2

import cv2
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as sktr

import skimage as sk
import skimage.transform as sktransform
from skimage.feature import canny
import skimage.io as skio
import time
import scipy

# global variables for drawing on mask
from skimage.transform import SimilarityTransform, warp

drawing = False
polygon = False
centerMode = False
contours = []
polygon_center = None
img = None

def create_mask(imname):
    masks_to_ret = {"centers":[], "contours":[], "offsets":[]}

    global drawing, polygon, contours, centerMode, polygon_center
    pressed_key = 0
    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        global drawing, centerMode, polygon, pressed_key
        if drawing == True and event == cv2.EVENT_MOUSEMOVE:
            cv2.circle(img,(x,y),10,(255,255,255),-1)
            cv2.circle(mask,(x,y),10,(255,255,255),-1)
        if polygon == True and event == cv2.EVENT_LBUTTONDOWN:
            contours.append([x,y])
            cv2.circle(img,(x,y),2,(255,255,255),-1)
        if centerMode == True and event == cv2.EVENT_LBUTTONDOWN:
            polygon_center = (x,y)
            print(polygon_center)
            cv2.circle(img, polygon_center, 3, (255, 0, 0), -1)
            centerMode = False

            masks_to_ret["centers"].append(polygon_center)
            masks_to_ret["contours"].append(contours)

    # Create a black image, a window and bind the function to window
    orig_img = cv2.imread(imname)
    reset_orig_img = orig_img[:]
    mask = np.zeros(orig_img.shape, np.uint8)
    img = np.array(orig_img[:])
    cv2.namedWindow('image')

    cv2.setMouseCallback('image',draw_circle)

    angle = 0
    delta_angle = 5
    resize_factor = 1.1
    total_resize = 1
    adjusted = False

    while(1):
        cv2.imshow('image',img)
        pressed_key = cv2.waitKey(20) & 0xFF

        """
        Commands:
        d: toggle drawing mode
        p: toggle polygon mode
        q: draw polygon once selected, and select center
        """

        if pressed_key == 27:
            break
        elif pressed_key == ord('d'):
            drawing = not drawing
            print("drawing status: ", drawing)
        elif pressed_key == ord('p'):
            polygon = not polygon
            print("polygon status: ", polygon)
        elif polygon == True and pressed_key == ord('q') and len(contours) > 2:
            contours = np.array(contours)
            cv2.fillPoly(img, pts=[contours], color = (255,255,255))
            cv2.fillPoly(mask, pts=[contours], color = (255,255,255))

            centerMode = True
            polygon = False
        elif pressed_key == ord('o'):
            # loop over the rotation angles again, this time ensuring
            # no part of the image is cut off
            angle = (angle + delta_angle) % 360
            adjusted = True
            print("Rotate")

        elif pressed_key == ord('i'):
            # loop over the rotation angles again, this time ensuring
            # no part of the image is cut off
            angle = (angle - delta_angle) % 360  
            adjusted = True
            print("Rotate")
        
        # Plus
        elif pressed_key == ord('='):
            total_resize = total_resize*resize_factor
            adjusted = True
            print("Resize up")

        # Minus
        elif pressed_key == ord('-'):
            total_resize = total_resize*(1/resize_factor)
            adjusted = True
            print("Resize down")
        

        elif pressed_key == ord('r'):
            img = np.array(reset_orig_img)
            contours = []
            masks_to_ret["centers"] = []
            masks_to_ret["contours"] = []

            centerMode = False
            polygon = False
            angle = 0
            total_resize = 1

            print("polygon status: False")

        # adjust
        if adjusted:
            rows,cols,_ = orig_img.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
            img = cv2.resize(orig_img, dsize=(0,0), fx=total_resize, fy=total_resize)
            img = cv2.warpAffine(img,M,(cols,rows))
            cv2.imshow('image', img)
            adjusted = False
            

    cv2.destroyAllWindows()
    name = imname.split('/')[-1]

    # store offsets to allow recreation of masks in target image
    for center_num in range(len(masks_to_ret["centers"])):
        offset = []
        center = masks_to_ret["centers"][center_num]
        for point in masks_to_ret["contours"][center_num]:
            xoffset = point[0] - center[0]
            yoffset = point[1] - center[1]

            offset.append([xoffset, yoffset])
        masks_to_ret["offsets"].append(offset)

    # adjust the output image
    rows,cols,_ = orig_img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    adj_orig_img = cv2.resize(reset_orig_img, dsize=(0,0), fx=total_resize, fy=total_resize)
    adj_orig_img = cv2.warpAffine(adj_orig_img,M,(cols,rows))
    
    return masks_to_ret, adj_orig_img

def paste_mask(im2name, masks_to_ret, im2=None):
    im2masks_to_ret = {"centers":[], "contours":[]}

    # mouse callback function
    def draw_circle(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            centernew = [x,y]
            new_contour = []
            for offsets in masks_to_ret["offsets"]:
                for point in offsets:
                    xnew = point[0] + centernew[0]
                    ynew = point[1] + centernew[1]
                    new_contour.append([xnew, ynew])
            new_contour= np.array(new_contour)
            im2masks_to_ret["centers"].append(centernew)
            im2masks_to_ret["contours"].append(new_contour)
            
            cv2.fillPoly(img, pts=[new_contour], color = (255,255,255))

    # Create a black image, a window and bind the function to window
    if type(im2) == type(None):
        orig_img = cv2.imread(im2name)#np.zeros((512,512,3), np.uint8)
    else:
        orig_img = np.array(im2)

    img = np.array(orig_img[:])
    cv2.namedWindow('image')
    cv2.resizeWindow('image', 600,600)
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        cv2.imshow('image',img)
        pressed_key = cv2.waitKey(20) & 0xFF

        if pressed_key == 27:
            break 
        if pressed_key == ord('r'):
            img = np.array(orig_img)
            im2masks_to_ret["centers"] = []
            im2masks_to_ret["contours"] = []

    return im2masks_to_ret, orig_img

# run with 2 image names to generate and save masks and new source image
def save_masks(im1name, im2name):
    masks_to_ret, source_im = create_mask(imname)
    im2masks_to_ret, target_im = paste_mask(im2name=im2name, masks_to_ret=masks_to_ret)
    # im1 is the source, im2 is the target
    source_mask = np.zeros((source_im.shape[0], source_im.shape[1], 3))
    target_mask = np.zeros((target_im.shape[0], target_im.shape[1], 3))
    cv2.fillPoly(source_mask, np.array([masks_to_ret["contours"][0]]), (255,255,255))
    cv2.fillPoly(target_mask, np.array([im2masks_to_ret["contours"][0]]), (255,255,255))

    name1 = im1name.split('/')[-1]
    name1 = name1[:-4]

    name2 = im2name.split('/')[-1]
    name2 = name2[:-4]

    source_mask = np.clip(sk.img_as_float(source_mask), -1, 1)
    target_mask = np.clip(sk.img_as_float(target_mask), -1, 1)
    source_im = np.clip(sk.img_as_float(source_im), -1, 1)
    source_im = np.dstack([source_im[:,:,2], source_im[:,:,1], source_im[:,:,0]])
    # judy: shall we provide it?
    offset =  np.array(-im2masks_to_ret['contours'][0][0]) + np.array(masks_to_ret['contours'][0][0])
    tform = SimilarityTransform(translation=offset)
    warped = warp(source_im, tform, output_shape=target_im.shape)

    skio.imsave(name1 + "_mask.png", source_mask)
    skio.imsave(name2 + "_mask.png",target_mask)
    skio.imsave(name1 + "_newsource.png", warped)
    return source_mask, target_mask, source_im

# Example usage
imname = "source_01.jpg"
im2name = "target_01.jpg"
save_masks(imname, im2name)
