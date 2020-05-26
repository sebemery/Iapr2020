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
from Load_image import load_image
from skimage.color import rgb2gray
from skimage.transform import resize
import pickle
from Train_data_knn import *
from Features_operators import *

def classify_operator(op) :
    
    # knn model for operators recognition
    model_operator = pickle.load(open('model_operators.sav', 'rb'))
    classes_knn = {0:'addition',1:'substraction',2:'multiplication'}
    
    # convert rgb operator in a gray scale image
    op = rgb2gray(op)
    
    # Resize in 40x40
    op = resize(op,(40,40), anti_aliasing = True)
    
    # Binarize 
    thresh = threshold_otsu(op)
    op = op<thresh
    
    # Get the features
    features = get_features(op,6,training = False)
    
    if (features[3] == 2.) :
        return 'equal'
    if(features[3] == 3.) :
        return 'division'
    if(features[3] == 1.) :
        features_knn = np.expand_dims(features[:3],axis = 0)
        return classes_knn[int(model_operator.predict(features_knn))]
        
        