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

def cross_validation_knn(nb_rotation ,rotate ,Nb_coeff , nb_neighbors) :
    
    # Load data
    operators = load_train_data_knn(nb_rotation,rotate,)
    
    # Get features
    features, targets = get_features(operators,Nb_coeff)
    # Only take Fourier Descriptors
    fetures_knn = features[:,:3]
    
    #split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(features_knn,targets, test_size=0.2, random_state=1)

    k_range = range(1,nb_neighbors)
    scores = {}
    scores_list = []
    for k in k_range :
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(X_train,y_train)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test,y_pred)
        scores_list.append(metrics.accuracy_score(y_test,y_pred))

    plt.plot(k_range,scores_list)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing accuracy')
    
    
def train_knn(features,targets,nb_neighbors) :
    
    # knn
    knn = KNeighborsClassifier(n_neighbors = nb_neighbors)
    knn.fit(features, targets)
    
    # save model
    filename = 'model_operators.sav'
    pickle.dump(knn, open(filename, 'wb'))
