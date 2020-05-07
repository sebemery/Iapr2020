import cv2
import os
import skimage.io
import numpy as np

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
    images_names = [nm for nm in os.listdir(images_path) if '.jpg' in nm]  # make sure to only load .png
    images_names.sort()  # sort file names
    ic = skimage.io.imread_collection([os.path.join(images_path, nm) for nm in images_names])
    images = skimage.io.concatenate_images(ic)
    
    return images, images_names
    