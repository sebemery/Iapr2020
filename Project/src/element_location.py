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
from skimage.segmentation import clear_border


def segmentation(image, arrow_bb):
    """
    returns a binary image 
                    - Input : RGB image
                    - Output : binary image
    """

    # crop robot from image
    minr_a, minc_a, maxr_a, maxc_a = arrow_bb
    test_im = image.copy()
    test_im[minr_a-30:maxr_a+50, minc_a-30:maxc_a+50] = 255
    
    # Change exposure to get a brighter image
    bright_im = skimage.exposure.adjust_gamma(test_im, gamma=3/5, gain=1)
    
    gray = skimage.color.rgb2gray(test_im)
    
    # Contrast stretching
    a, b = np.percentile(gray, (1, 70))
    img_rescale = exposure.rescale_intensity(gray, in_range=(a, b))
    
    # Multi thresholds to get digits and operators at once
    thresholds = threshold_multiotsu(img_rescale)
    
    # Using the threshold values, we generate the three regions.
    regions = np.digitize(gray, bins=thresholds)
    
    # Binarization, discriminate background from foreground
    binarized = regions == 0
    
    # Morphological operation
    close = morphology.binary_dilation(binarized, skimage.morphology.selem.rectangle(1, 3))
    
    return close
    
def Overlap(l1, r1, l2, r2):
    """
    This function check if two rectangles overlap
                    - Input : left upper corner and right bottom corner or both rectangle
                    - Output : True if overlap, false otherwise
    """
    if(l1[0] >= r2[0] or l2[0] >= r1[0]):
        return False
    
    if(l1[1] >= r2[1] or l2[1] >= r1[1]):
        return False
  
    return True

def get_object_bb(binarized_im):
    """
    This function returns the bounding box for labels in a binary image
                    - Input : binary image
                    - Output : boxes and indice of last label (used for checkboxes function)
    """

    # Get labels
    cleared = clear_border(binarized_im)
    label_image = label(cleared, background = 0, connectivity = 2)
    boxes = {}
    ind = 0

    # For all labels get bounding box depending and size of label
    for region in regionprops(label_image):
        minr, minc, maxr, maxc = region.bbox
        # take regions with large enough areas
        if (region.area >= 50) and (maxr-minr) < 40 and (maxc-minc) > 10:
            ind += 1
            boxes['label_'+str(ind)] = [minr, minc, maxr, maxc]    
    return boxes, ind

def checkboxes(boxes, ind):
    """
    This function check overlap between label, merge them if touching and update the object list
                    - Input : dict containing the bb of all labels and indice of last label
                    - Output : dict of bb without any overlap 

    """
    overlap = []
    new_boxes = {}
    for box_i in boxes:
        minr_i, minc_i, maxr_i, maxc_i = boxes[box_i]
        x_i, y_i = minc_i+(maxc_i-minc_i)/2, minr_i+(maxr_i-minr_i)/2
        for box_j in boxes:
            if box_i != box_j:
                minr_j, minc_j, maxr_j, maxc_j = boxes[box_j]
                x_j, y_j = minc_j+(maxc_j-minc_j)/2, minr_j+(maxr_j-minr_j)/2
                if Overlap([x_i-25, y_i-25], [x_i+25, y_i+25], [x_j-25, y_j-25], [x_j+25, y_j+25]):
                    if box_i not in overlap and box_j not in overlap:
                        overlap += [box_i, box_j]
                        ind += 1
                        new_boxes['label_'+str(ind)] = (minr_i+minr_j)/2, (minc_i+minc_j)/2, (maxr_i+maxr_j)/2, (maxc_i+maxc_j)/2
    
    for del_box in overlap:
        del boxes[del_box]
        
    boxes.update(new_boxes)
    
    return overlap

def crop_image(image, bbox):
    """
    This function returns a crop of an image given the crop position
    """
    crop_im = image[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
    return crop_im