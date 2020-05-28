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

def classify_operator(op,model_operator) :
    
    # knn classes
    classes_knn = {0:'addition', 1:'substraction', 2:'multiplication'}
    
    # convert rgb operator in a gray scale image
    op = rgb2gray(op)
    
    # Resize in 40x40
    op = resize(op, (40,40), anti_aliasing = True)
    
    # Binarize 
    thresh = threshold_otsu(op)
    op = op<thresh
    
    # Get the features
    features = get_features(op, 6, training = False)
    
    if (features[4] == 2.) :
        return 'equal'
    if(features[4] == 3.) :
        return 'division'
    if(features[4] == 1.) :
        features_knn = np.expand_dims(features[:4], axis = 0)
        return classes_knn[int(model_operator.predict(features_knn))]
        
        
