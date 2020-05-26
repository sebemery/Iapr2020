import os
import math
import numpy as np
import skimage.io
from scipy import ndimage as ndi

def load_image(path,image_name) :
    
    # Load image and store the image in an numpy array
    data_path = os.path.join(path)
    image = skimage.io.imread(os.path.join(data_path, image_name))
    
    return image
