import os
import math
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.measure import find_contours



def Fourier_descriptors(operator, Nb_coeff) :
    """
    Calculate the DFT of the operator's contour 
    Return the first N coefficient and the number of disjoint contours 
    """
    # get the contours
    contour = find_contours(operator,0.2)
    nb_contours = len(contour)
    if (nb_contours == 1) :
        contour = np.squeeze(np.asarray(contour))
        contour_complex = np.empty(contour.shape[0], dtype=complex)
        contour_complex.real = contour[:, 0]
        contour_complex.imag = contour[:, 1]
        fourier_result = np.fft.fft(contour_complex)
        DFT = np.fft.fft(contour_complex)
    if (nb_contours == 2) :
        contour_concatenated = np.concatenate((contour[0],contour[1]), axis=0)
        contour_complex = np.empty(contour_concatenated.shape[0], dtype=complex)
        contour_complex.real = contour_concatenated[:, 1]
        contour_complex.imag = contour_concatenated[:, 0]
        DFT = np.fft.fft(contour_complex)
    if (nb_contours == 3) :
        contour_concatenated = np.concatenate((contour[0],contour[1],contour[2]), axis=0)
        contour_complex = np.empty(contour_concatenated.shape[0], dtype=complex)
        contour_complex.real = contour_concatenated[:, 1]
        contour_complex.imag = contour_concatenated[:, 0]
        DFT = np.fft.fft(contour_complex)
    
    return DFT[:Nb_coeff], nb_contours

def get_features(operators, Nb_coeff, training=False) :
    """
    Input : (Nx5x40x40) arrays of operators 
    Output : (5Nx4) arrays of features and (5N) arrays of target
            Features : First three features are ratio of the amplitude of fourier descriptor -> [A1/A5,A2/A5,A3/A4]
                       Last features is the number of disjoint contour -> plus,minus,mul : 1, equal : 2, div :3
    """
    if(training == True) :
        # initialize the arrrays
        features = np.empty((operators.shape[0],operators.shape[1],5))
        targets = np.empty((operators.shape[0],operators.shape[1]))
        #Loop over the array of operators to get the features and targets
        for N in range(operators.shape[0]):
            for idx,op in enumerate(operators[N]):
                coeff,nb_contours = Fourier_descriptors(op,Nb_coeff)
                A1 = np.sqrt(coeff[1].real**2+coeff[1].imag**2)
                A2 = np.sqrt(coeff[2].real**2+coeff[2].imag**2)
                A3 = np.sqrt(coeff[3].real**2+coeff[3].imag**2)
                A4 = np.sqrt(coeff[4].real**2+coeff[4].imag**2)
                A5 = np.sqrt(coeff[5].real**2+coeff[5].imag**2)
                # compute ratios of Fourier descriptors -> scale invariant
                coord_x = (A1/A5)
                coord_y = (A2/A5)
                coord_z = (A3/A5)
                coord_w = (A4/A5)
                # concatenate the features 
                features[N,idx] = np.array([coord_x,coord_y,coord_z,coord_w,nb_contours])
                targets[N,idx] = idx
        # reshape the arrays for the classification algorithm
        features = np.reshape(features,(operators.shape[0]*operators.shape[1],5))
        targets = np.reshape(targets,(operators.shape[0]*operators.shape[1]))

        return features,targets
        
    else :
        coeff,nb_contours = Fourier_descriptors(operators,Nb_coeff)
        A1 = np.sqrt(coeff[1].real**2+coeff[1].imag**2)
        A2 = np.sqrt(coeff[2].real**2+coeff[2].imag**2)
        A3 = np.sqrt(coeff[3].real**2+coeff[3].imag**2)
        A4 = np.sqrt(coeff[4].real**2+coeff[4].imag**2)
        A5 = np.sqrt(coeff[5].real**2+coeff[5].imag**2)
        # compute ratios of Fourier descriptors -> scale invariant
        coord_x = (A1/A5)
        coord_y = (A2/A5)
        coord_z = (A3/A5)
        coord_w = (A4/A5)
        # concatenate the features 
        features = np.array([coord_x,coord_y,coord_z,coord_w,nb_contours])
        
        return features