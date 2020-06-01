import tarfile
import os
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu,median
from skimage.measure import moments
from skimage.transform import warp,AffineTransform
from skimage.transform import resize
from skimage.io import imsave
from Load_image import load_image



def generate_masks(image_name = 'original_operators.png') :
    """
     A function used to generate the binary mask of operators used to train the operator classifier 

     Input (by default) : original_operators.png
     Output : Binarized and non binarized masks of each operator (40x40) -> mul_bin.png and  mul.png 
    """

    # Load image and store the image in an  numpy array
    operators = load_image('../data',image_name)

    # Transform the RGB image of operators in a gray scale image
    operators_gray = rgb2gray(operators)

    # cut manually the operators in five arrays
    add = operators_gray[:,0:349]
    equal = operators_gray[:,349:697]
    minus = operators_gray[:,697:1045]
    div = operators_gray[:,1045:1393]
    mul = operators_gray[:,1393:1736]

    # Binarization based on Otsu threshold
    ## add
    add_thresh = threshold_otsu(add)
    add_binarized = add < add_thresh
    ## equal
    equal_thresh = threshold_otsu(equal)
    equal_binarized = equal < equal_thresh
    ## minus
    minus_thresh = threshold_otsu(minus)
    minus_binarized = minus < minus_thresh
    ## div
    div_thresh = threshold_otsu(div)
    div_binarized = div < div_thresh
    ## mul
    mul_thresh = threshold_otsu(mul)
    mul_binarized = mul < mul_thresh
    
    # Calculate the centroid of the objects
    ## add
    M_add = moments(add_binarized)
    centroid_add = (M_add[1, 0] / M_add[0, 0], M_add[0, 1] / M_add[0, 0])
    ## equal
    M_equal = moments(equal_binarized)
    centroid_equal = (M_equal[1, 0] / M_equal[0, 0], M_equal[0, 1] / M_equal[0, 0])
    ## minus
    M_minus = moments(minus_binarized)
    centroid_minus = (M_minus[1, 0] / M_minus[0, 0], M_minus[0, 1] / M_minus[0, 0])
    ## div
    M_div = moments(div_binarized)
    centroid_div = (M_div[1, 0] / M_div[0, 0], M_div[0, 1] / M_div[0, 0])
    ## mul
    M_mul = moments(mul_binarized)
    centroid_mul = (M_mul[1, 0] / M_mul[0, 0], M_mul[0, 1] / M_mul[0, 0])
    
    # Center the operator
    ## add 
    tadd = AffineTransform(translation=(-18, 2))
    add_centered = warp(add_binarized,tadd)
    ## equal
    tequal = AffineTransform(translation=(-27, -2))
    equal_centered = warp(equal_binarized,tequal)
    ## minus
    tminus = AffineTransform(translation=(-3, -7))
    minus_centered = warp(minus_binarized,tminus)
    ## div
    tdiv = AffineTransform(translation=(18, -15))
    div_centered = warp(div_binarized,tdiv)
    ## mul 
    tmul = AffineTransform(translation=(26, 1))
    mul_centered = warp(mul_binarized,tmul)
    
    
    # Resize the mask in (40x40)
    add_resized = resize(add_centered,(40,40), anti_aliasing = True)
    equal_resized = resize(equal_centered,(40,40), anti_aliasing = True)
    minus_resized = resize(minus_centered,(40,40), anti_aliasing = True)
    div_resized = resize(div_centered,(40,40), anti_aliasing = True)
    mul_resized = resize(mul_centered,(40,40), anti_aliasing = True)
    
    # Binarize the mask after interpolation
    ## add
    add_thresh = threshold_otsu(add_resized)
    add_resized_binarized = add_resized>add_thresh
    add_resized_binarized = add_resized_binarized.astype(float)
    ## equal
    equal_thresh = threshold_otsu(equal_resized)
    equal_resized_binarized = equal_resized>equal_thresh
    equal_resized_binarized = equal_resized_binarized.astype(float)
    ## minus
    minus_thresh = threshold_otsu(minus_resized)
    minus_resized_binarized = minus_resized>minus_thresh
    minus_resized_binarized = minus_resized_binarized.astype(float)
    ## div
    div_thresh = threshold_otsu(div_resized)
    div_resized_binarized = div_resized>div_thresh
    div_resized_binarized = div_resized_binarized.astype(float)
    ## mul
    mul_thresh = threshold_otsu(mul_resized)
    mul_resized_binarized = mul_resized>mul_thresh
    mul_resized_binarized = mul_resized_binarized.astype(float)
    
    # Save mask and binarized mask for training
    ## Non binarized
    imsave('Masks/mul.png', mul_resized)
    imsave('Masks/plus.png', add_resized)
    imsave('Masks/equal.png', equal_resized)
    imsave('Masks/minus.png', minus_resized)
    imsave('Masks/div.png', div_resized)
    ## Binarized
    imsave('Masks/mul_bin.png', mul_resized_binarized)
    imsave('Masks/plus_bin.png', add_resized_binarized)
    imsave('Masks/equal_bin.png', equal_resized_binarized)
    imsave('Masks/minus_bin.png', minus_resized_binarized)
    imsave('Masks/div_bin.png', div_resized_binarized)
