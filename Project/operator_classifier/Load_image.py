import os
import math
import numpy as np
import skimage.io
from scipy import ndimage as ndi

def load_image(path,image_name) :
    """
    Load  an image and store it in an numpy array given a path and the file name
    """
    data_path = os.path.join(path)
    image = skimage.io.imread(os.path.join(data_path, image_name))
    
    return image
