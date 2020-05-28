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

def extract_images(data_path='../data/robot_parcours_1.avi'):
    video = cv2.VideoCapture(data_path)

    if not os.path.exists('../input'):
        os.makedirs('../input')
    
    print ('> Extracting images from video')
    
    currentframe = 0
    while(True):
        # Extract images
        ret, frame = video.read()
        # end of frames
        if ret: 
            # Saves images
            if currentframe < 10:
                name = '../input/frame' + '0' + str(currentframe) + '.jpg'
            else:
                name = '../input/frame' + str(currentframe) + '.jpg'
            
            cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break
    
    print ('done')
    
    video.release()
    cv2.destroyAllWindows()
    
def load_data(plot_images=False): 
    images_path = '../input/'
    images_names = [nm for nm in os.listdir(images_path) if '.jpg' in nm]  # make sure to only load .jpg
    images_names.sort()  # sort file names
    ic = skimage.io.imread_collection([os.path.join(images_path, nm) for nm in images_names])
    images = skimage.io.concatenate_images(ic)
    
    if plot_images:
        # plot images
        fig, axes = plt.subplots(5, int(len(images)/5)+1, figsize=(24, 8))
        for ax, im, nm in zip(axes.ravel(), images, images_names):
            ax.imshow(im)
            ax.axis('off')
            ax.set_title(nm)   
        for ax in axes.ravel()[len(images):]:
            fig.delaxes(ax)
        plt.show()

    return images, images_names

def frames_to_video(inputpath, outputpath, fps):    
    image_array = []
    files = [nm for nm in os.listdir(inputpath) if '.jpg' in nm]  # make sure to only load .jpg
    files.sort()
    for i in range(len(files)):
        img = cv2.imread(inputpath + files[i])
        size =  (img.shape[1], img.shape[0])
        img = cv2.resize(img,size)
        image_array.append(img)
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter(outputpath,fourcc, fps, size)
    for i in range(len(image_array)):
        out.write(image_array[i])
    out.release()