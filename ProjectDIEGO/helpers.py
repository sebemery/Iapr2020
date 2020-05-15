import cv2
import os
import skimage.io
import numpy as np

import skimage
from skimage.measure import label, regionprops, find_contours
from skimage.filters import threshold_minimum, threshold_multiotsu
from skimage import morphology
from skimage import exposure

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
    return image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def frames_to_video(inputpath, outputpath, fps):    
    image_array = []
    files = [nm for nm in os.listdir(inputpath) if '.jpg' in nm]  # make sure to only load .jpg
    files.sort()
    for i in range(len(files)):
        img = cv2.imread(inputpath + files[i])
        size =  (img.shape[1],img.shape[0])
        img = cv2.resize(img,size)
        image_array.append(img)
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter(outputpath,fourcc, fps, size)
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()