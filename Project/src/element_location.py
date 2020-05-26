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


def segmentation(image, arrow_bb):
    minr_a, minc_a, maxr_a, maxc_a = arrow_bb
    test_im = image.copy()
    test_im[minr_a-30:maxr_a+50, minc_a-30:maxc_a+50] = 255
    
    gray = skimage.color.rgb2gray(test_im)
    
    # Contrast stretching
    a, b = np.percentile(gray, (1, 70))
    img_rescale = exposure.rescale_intensity(gray, in_range=(a, b))
    
    # Threshols
    thresholds = threshold_multiotsu(img_rescale)
    
    # Using the threshold values, we generate the three regions.
    regions = np.digitize(gray, bins=thresholds)
    
    # Binarization 
    binarized = regions == 0
    
    # Morphological operation
    close = morphology.binary_dilation(binarized, skimage.morphology.selem.rectangle(1, 3))
    
    return close

def contours(binarized_im):
    contour = find_contours(binarized_im, 0)
    
def get_object_bb(binarized_im):
    label_image = skimage.measure.label(binarized_im, background = 0, connectivity = 2)
    freq = np.bincount(label_image.ravel())
    ii = np.nonzero(freq)[0]
    
    for i in range(len(np.bincount(label_image.ravel()))-1):
        if (abs(freq[i+1]-freq[i])<10):
            
            idx = np.argwhere(label_image == ii[i+1])
            label_image[idx[:,0],idx[:,1]] = ii[i]
        
    for i in range(len(np.bincount(label_image.ravel()))-2) :
        if (abs(freq[i+2]-freq[i])<5 and abs(freq[i+1]-freq[i])<75) :
            idx2 = np.argwhere(label_image == ii[i+2])
            idx1 = np.argwhere(label_image == ii[i+1])
            label_image[idx2[:,0],idx2[:,1]] = ii[i]
            label_image[idx1[:,0],idx1[:,1]] = ii[i]

    boxes = {}
    label = 0
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        # take regions with large enough areas
        if (region.area >= 50) and (maxr-minr) < 40:
            label += 1
            boxes['label_'+str(label)] = [minr, minc, maxr, maxc]    
    
    return boxes

def Overlap(l1, r1, l2, r2):
    if(l1[0] >= r2[0] or l2[0] >= r1[0]):
        return False
    
    if(l1[1] >= r2[1] or l2[1] >= r1[1]):
        return False
  
    return True

def crop_image(image, bbox):
    
    crop_im = image[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
    gray = skimage.color.rgb2gray(crop_im)
    a, b = np.percentile(gray, (1, 70))
    img_rescale = exposure.rescale_intensity(gray, in_range=(a, b))
    thresholds = threshold_multiotsu(img_rescale)
    regions = np.digitize(gray, bins=thresholds)
    binarized = regions == 0 
    return rescale(binarized.astype(float), 40/50 , anti_aliasing=False)