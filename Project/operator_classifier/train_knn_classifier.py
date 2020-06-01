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
import pickle
from training_data import load_train_data_knn
from features import get_features

from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
    
    
def train_knn(features,targets,nb_neighbors) :
    """
     A function to train the KNN classifier and save it

     Input : Arrays of features and corresponding target plus the number of neighbors to consider in the model
     Ouput : Classifier saved in 'model_operators.sav' using pickle 
    """
    
    # knn
    knn = KNeighborsClassifier(n_neighbors = nb_neighbors)
    knn.fit(features, targets)
    
    # save model
    filename = 'model/model_operators.sav'
    pickle.dump(knn, open(filename, 'wb'))



if __name__ == "__main__":

    """
        An executable to cross validate the number of neighbors to use

        It use a grid search on odd neigbors from 1 to 25 by a 10-fold cross validation
            -> print the array of accuracy [0-1]        
    """

    # Load data
    operators = load_train_data_knn(360,True)
    
    # Get features
    features, targets = get_features(operators,6,True)
    # Only take Fourier Descriptors
    features_knn = features[:,:4]

    print(features_knn.shape)
    print(targets.shape)

    #create new a knn model
    knn_GS = KNeighborsClassifier()
    
    #create a dictionary of all values we want to test for n_neighbors
    param_grid = {'n_neighbors': [1,3,5,7,9,11,13,15,17,19,21,23,25]}
    
    #use gridsearch to test all values for n_neighbors
    knn_gscv = GridSearchCV(knn_GS, param_grid, cv=10)
    
    #fit model to data
    knn_gscv.fit(features_knn, targets)

    # check result
    print(knn_gscv.cv_results_['mean_test_score'])
    

    

    
