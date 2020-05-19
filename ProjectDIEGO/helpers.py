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
    
def load_data(plot_images=False): 
    images_path = './images/'
    images_names = [nm for nm in os.listdir(images_path) if '.jpg' in nm]  # make sure to only load .jpg
    images_names.sort()  # sort file names
    ic = skimage.io.imread_collection([os.path.join(images_path, nm) for nm in images_names])
    images = skimage.io.concatenate_images(ic)
    
    if plot_images == True:
        # plot images
        fig, axes = plt.subplots(5, int(len(images)/5)+1, figsize=(24, 8))
        for ax, im, nm in zip(axes.ravel(), images, images_names):
            ax.imshow(im)
            ax.axis('off')
            ax.set_title(nm)   
        for ax in axes.ravel()[len(images):]:
            fig.delaxes(ax)
    
    return images, images_names

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
    return rescale(binarized.astype(float), 40/50, anti_aliasing=False)[:,:]

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
    
def get_features(operator, nb_coeff) :
    """
    Input : (Nx5x40x40) arrays of operators 
    Output : (5Nx4) arrays of features and (5N) arrays of target
            Features : First three features are ratio of the amplitude of fourier descriptor -> [A1/A2, A3/A2, A4/A2]
                       Last features is the number of disjoint contour -> plus, minus, mul : 1, equal : 2, div :3
    """    
    # compute the features -> amplitude and first fourier descriptors excluded -> invariance in translation and rotation
    coeff, nb_contours = fourier_descriptors(operator, nb_coeff)
    A1 = np.sqrt(coeff[1].real**2+coeff[1].imag**2)
    A2 = np.sqrt(coeff[3].real**2+coeff[3].imag**2)
    A3 = np.sqrt(coeff[4].real**2+coeff[4].imag**2)
    A4 = np.sqrt(coeff[2].real**2+coeff[2].imag**2)
    A5 = np.sqrt(coeff[5].real**2+coeff[5].imag**2)
    # compute ratios of Fourier descriptors -> scale invariant
    coord_x = (A1/A1)
    coord_y = (A2/A1)
    coord_z = (A3/A1)
    coord_w = (A4/A1)
    coord_v = (A5/A1)
    # concatenate the features 
    features = np.array([coord_x,coord_y,coord_z,coord_w,coord_v,nb_contours])

    return features
    
def fourier_descriptors(operators, nb_coeff):
    """
    Calculate the DFT of the operator's contour 
    Return the first N coefficient and the number of disjoint contours 
    """
    # get the contours
    contour = find_contours(operators,0.2)
    nb_contours = len(contour)
    if (nb_contours == 1) :
        contour = np.squeeze(np.asarray(contour))
        contour_complex = np.empty(contour.shape[0], dtype=complex)
        contour_complex.real = contour[:, 0]
        contour_complex.imag = contour[:, 1]
        fourier_result = np.fft.fft(contour_complex)
        DFT = np.fft.fft(contour_complex)
    if (nb_contours == 2) :
        contour_concatenated = np.concatenate((contour[0],contour[1]), axis=0)
        contour_complex = np.empty(contour_concatenated.shape[0], dtype=complex)
        contour_complex.real = contour_concatenated[:, 1]
        contour_complex.imag = contour_concatenated[:, 0]
        DFT = np.fft.fft(contour_complex)
    if (nb_contours == 3) :
        contour_concatenated = np.concatenate((contour[0],contour[1],contour[2]), axis=0)
        contour_complex = np.empty(contour_concatenated.shape[0], dtype=complex)
        contour_complex.real = contour_concatenated[:, 1]
        contour_complex.imag = contour_concatenated[:, 0]
        DFT = np.fft.fft(contour_complex)
    
    return DFT[:nb_coeff], nb_contours

#def evaluate_operation(string):
#    if isinstance(operation, int):
#        return result + operation 
#    elif isinstance(operation, str):
#        if operation == '+':
#            return
#        elif operation == '-':
#        elif operation == '/':
#        elif operation == '*':
#
#        classes = {0:'+', 1:'=', 2:'-', 3:'/', 4:'*'}
#     
    