import cv2
import os
import skimage.io
import numpy as np

import skimage
from skimage.measure import label, regionprops
from skimage.filters import threshold_minimum
from skimage import morphology

def extract_images(data_path='./data/robot_parcours_1.avi'):
    video = cv2.VideoCapture(data_path)

    if not os.path.exists('images'):
        os.makedirs('images')
    
    print ('> Extracting images from video')
    
    currentframe = 0
    while(True):
        # Extract images
        ret, frame = video.read()
        # end of frames
        if ret: 
            # Saves images
            if currentframe < 10:
                name = './images/frame' + '0' + str(currentframe) + '.jpg'
            else:
                name = './images/frame' + str(currentframe) + '.jpg'
            
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
    
    print ('done')
    
    video.release()
    cv2.destroyAllWindows()
    
def load_data(): 
    images_path = './images/'
    images_names = [nm for nm in os.listdir(images_path) if '.jpg' in nm]  # make sure to only load .jpg
    images_names.sort()  # sort file names
    ic = skimage.io.imread_collection([os.path.join(images_path, nm) for nm in images_names])
    images = skimage.io.concatenate_images(ic)
    
    return images, images_names

def biggest_label(properties):
    # assigns to max_lab the biggest labeled object on the image
    max_lab = properties[0]
    for lab in properties:
        if lab.area > max_lab.area:
            max_lab = lab
    return max_lab

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
    