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

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

def rotate_operators(operators, nb_rotations) :
    """
    Input : (1xNb_operatorsx40x40) array of operators concatenated
    Output : (NxNb_opratorsx40x40)  N arrays of five operators where they have been rotated N times to reach 360
    """
    #calculate the array of angles of rotations
    angle = 360/nb_rotations
    angles = np.linspace(angle,360-angle,nb_rotations-1)
    
    for a in angles :
        operators_rotated = np.empty((1,operators.shape[1],40,40))
        for idx,op in enumerate(operators[0]) :
            # rotate operators with bi-cubic interpolation
            operators_rotated[0,idx] = rotate(op,a,order =5)
            # binarize the resulting rotated opearators
            thresh = threshold_otsu(operators_rotated[0,idx])
            operators_rotated[0,idx] = operators_rotated[0,idx]>thresh
            operators_rotated[0,idx] = operators_rotated[0,idx].astype(float)
            
        # concatenate the five operators rotated by a certain angle to the array
        operators = np.concatenate([operators,operators_rotated],axis=0)
    return operators

def load_train_data_knn(nb_rotation,rotate=False) :
    """
    Load the three operators mask for knn (40x40) -> plus,minus,mul
    Output : (Nx3x40x40) array of operators :
                If no operators rotation : N=1 (1x3x40x40) -> [[plus,minus,mul]]
                If operators are rotated : (Nx3x40x40) where N correspond to the number of rotation of each operator
                                            N = 360/ nb_rotation
    """
    
    # Load operators
    plus = load_image('plus_bin.png')
    minus = load_image( 'minus_bin.png')
    mul = load_image('mul_bin.png')
    
    # concatenate operators
    
    plus = np.expand_dims(plus,axis = 0)
    minus = np.expand_dims(minus,axis = 0)
    mul = np.expand_dims(mul,axis = 0)
    
    operators = np.concatenate([plus,minus,mul], axis = 0)
    operators = np.expand_dims(operators,axis = 0)
    
    #rotation of the operators if true
    if (rotate == True) :
        operators = rotate_operators(operators,nb_rotation)
        
    return operators