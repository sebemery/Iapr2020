import cv2
import os
import skimage.io
import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage.measure import label, regionprops, find_contours
from skimage.filters import threshold_minimum, threshold_multiotsu
from skimage.transform import rescale
from skimage import morphology
from skimage import exposure

def biggest_label(properties):
    # assigns to max_lab the biggest labeled object on the image
    max_lab = properties[0]
    for lab in properties:
        if lab.area > max_lab.area:
            max_lab = lab
    return max_lab

def get_arrow_bb_test(img):
    mask_red = img[:,:,2] > 90
    test_mask = np.copy(img)
    test_mask[mask_red] = 0
    
    gray = skimage.color.rgb2gray(test_mask[:,:,0])
    
    threshold = threshold_minimum(gray)
    binary = gray > threshold
    close = morphology.binary_closing(binary, skimage.morphology.selem.disk(3))
    
    labels = label(close)
    biggest_lab = biggest_label(regionprops(labels))
    
    return biggest_lab.bbox, biggest_lab.orientation, biggest_lab.minor_axis_length, biggest_lab.major_axis_length, biggest_lab.centroid

def get_arrow_bb(img):
    mask_red = img[:,:,2] > 90
    test_mask = np.copy(img)
    test_mask[mask_red] = 0
    
    gray = skimage.color.rgb2gray(test_mask[:,:,0])
    
    threshold = threshold_minimum(gray)
    binary = gray > threshold
    close = morphology.binary_closing(binary, skimage.morphology.selem.disk(3))
    
    labels = label(close)
    biggest_lab = biggest_label(regionprops(labels))
    
    return biggest_lab.bbox