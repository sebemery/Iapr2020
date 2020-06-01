from os.path import dirname, join as pjoin
import scipy.io as sio
import numpy as np
import random
import os
import skimage.io
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from skimage.transform import rescale
from skimage.transform import rotate
from skimage.color import rgb2gray
import cv2

def get_digits_images(path, show_imgs=False):
    '''
    loads crop images of digits of the training video + creates targets 
    '''
    names = [nm for nm in os.listdir(path) if '.jpg' in nm]  # make sure to only load .jpg
    names.sort()  # sort file names
    ic_digits = skimage.io.imread_collection([os.path.join(path, nm) for nm in names])
    images = skimage.io.concatenate_images(ic_digits)
    
    target = [3,2,7,2]
    
    if show_imgs:
        fig, axes = plt.subplots(1, images.shape[0], figsize=(16, 4))
        for ax, im, nm in zip(axes.ravel(), images, names):
            ax.imshow(im)
            ax.axis('off')
            ax.set_title(nm)
               
    return images, target  

def togray_imgs(imgs):
    '''
    applies rgb2gray() to list of images
    '''
    images = []
    for im in imgs[:,:,:,0]:
            im = rgb2gray(im)
            images.append(im)
    return images

def binarize_imgs(imgs):
    '''
    binarizes list of images using Otsu threshold
    '''
    images = []
    for im in imgs:
        thresh = threshold_otsu(im)
        im = im < thresh
        im = im.astype(int)
        images.append(im)
    return images

def binarize_imgs_2(imgs):
    '''
    binarizes list of images using Otsu threshold applying threshold in other sense 
    than binarize_imgs() to get black background
    '''
    images = []
    for im in imgs:
        thresh = threshold_otsu(im)
        im = im > thresh
        im = im.astype(int)
        images.append(im)
    return images

def rotate_one_digit(imgs, nb, digit):
    '''
    creates nb random rotations of one digit + associated targets
    '''
    dataset = []
    targets = [digit] * nb
    for k in range(nb):
        dataset.append(rotate(imgs[0], np.random.randint(360), cval=0))
    return dataset, targets

def save_results(imgs, target, path_imgs, path_target):
    '''
    save rotated images + targets 
    '''
    i=0
    # save targets
    np.save(path_target, target)
    # save rotated images
    for im in imgs:
        name = path_imgs + str(i+1) + '.jpg'
        plt.imsave(name, im, cmap='gray')
        i+=1

def perform_rotation(path_input, path_imgs, path_target):
    '''
    performs processing, rotations and saves images + targets
    '''
    # get images, rgb2gray and binarize
    imgs, target = get_digits_images(path_input)
    imgs = togray_imgs(imgs)
    imgs = binarize_imgs(imgs)

    # create list for each digit and randomly rotate it 500 times
    imgs2_1 = []
    imgs2_1.append(imgs[1])
    imgs2_1, target2_1 = rotate_one_digit(imgs2_1, 500, 2)
    imgs2_2 = []
    imgs2_2.append(imgs[3])
    imgs2_2, target2_2 = rotate_one_digit(imgs2_2, 500, 2)
    imgs3 = []
    imgs3.append(imgs[0])
    imgs3, target3 = rotate_one_digit(imgs3, 500, 3)
    imgs7 = []
    imgs7.append(imgs[2])
    imgs7, target7 = rotate_one_digit(imgs7, 500, 7)

    # concatenate all lists
    imgs = imgs2_1+imgs2_2+imgs3+imgs7
    target = target2_1+target2_2+target3+target7

    # binarize again to get black background
    imgs = binarize_imgs_2(imgs)

    # save obtained images + targets  
    save_results(imgs, target, path_imgs, path_target)




