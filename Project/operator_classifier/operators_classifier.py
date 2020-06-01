import os
import math
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import find_contours
from skimage.transform import rotate
from skimage.morphology import disk,square,closing
from scipy import ndimage as ndi
from operator_classifier.Load_image import load_image
from skimage.color import rgb2gray
from skimage.transform import resize
import pickle
from operator_classifier.features import *

def classify_operator(mask_op,model_operator) :
    """
     Function that classify an rgb operator mask given the saved knn model

     Input : Rgb mask of the operator to classify (any size) and the knn classifier saved
     Ouptut : Operator's name string

     Principle :
         - Take the rgb mask of any size and convert it in a grayscale image
         - Rescale it to (40x40)
         - Binarize it
         - Get the features
         - Check the number of disjoints contour by checking the last feature
             -> if 1 it can be an addition, substraction or multiplication
                 -> call the knn classifier on the remaining features (fourier descriptors) to discriminate them
             -> if 2 it is the equal operator
             -> if 3 it is the division operatorr
    """

    # knn classes
    classes_knn = {0:'addition', 1:'substraction', 2:'multiplication'}
    
    # convert rgb operator in a gray scale image
    mask_op = rgb2gray(mask_op)
    
    # Resize in 40x40
    mask_op = resize(mask_op, (40,40), anti_aliasing = True)
    
    # Binarize 
    thresh = threshold_otsu(mask_op)
    mask_op = mask_op<thresh
    
    # Get the features
    features = get_features(mask_op, 6, training = False)
    
    if features[4] == 2.:
        return 'equal'
    if features[4] == 3.:
        return 'division'
    if features[4] == 1.:
        features_knn = np.expand_dims(features[:4], axis = 0)
        return classes_knn[int(model_operator.predict(features_knn))]
        
        
