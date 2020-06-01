import torch
import scipy.io as sio
import numpy as np
import random
import os
import skimage.io
from skimage.filters import threshold_otsu
from skimage.transform import rescale
from skimage.transform import rotate
from skimage.color import rgb2gray
from sklearn.utils import shuffle
import zipfile
from generate_rotated_imgs import perform_rotation

def loadmat(filename):
    '''
    loads matfile into dictionnaries + checks if entries are matlab object
    '''
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        


def _todict(matobj):
    '''
    constructs nested dictionaries from matobjects
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def load_data(path):
    '''
    loads affNIST 'normal' 40x40 images
    '''
    dataset = loadmat(path)
    target = dataset['affNISTdata']['label_int']
    data = dataset['affNISTdata']['image'].transpose()

    data = np.array(data).reshape((-1, 1, 40, 40)).astype(np.float32)
    
    for i in range(len(target)):
        thresh = threshold_otsu(data[i][0])
        data[i][0] = data[i][0] > thresh
    return data, target


def binarize(im):
    '''
    binarize the image using Otsu threshold
    '''
    thresh = threshold_otsu(im)
    return im > thresh


def random_rotations(imgs, imgs_labels, nb):
    '''
    generate nb random rotations of imgs (and return also matching labels (=targets))
    '''
    dataset = []
    labels =[]
    for k in range(nb):
        ind = np.random.randint(len(imgs))
        dataset.append(binarize(rotate(imgs[ind], np.random.randint(360), cval=0)).astype(int))
        labels.append(imgs_labels[ind])
    return np.array(dataset), np.array(labels)


def take_n_digit(data, data_target, digit, nb):
    '''
    for each digits of original data, generates nb random samples
    '''
    ind = []
    for i in range(len(data_target)):
        if(data_target[i]==digit):
            ind.append(i)
    new_data = data[ind]
    new_data_target = data_target[ind]
    
    random_ind = random.sample(range(1, len(new_data_target)), nb)
    new_data = new_data[random_ind,:,:,:]
    new_data_target = new_data_target[random_ind]
    return new_data, new_data_target


def create_uniform_dataset(train_input, train_target, test_input, test_target, nb_train, nb_test):
    '''
    generate uniform train and test sets with same number of each digit using take_n_digit function
    '''
    tmp_train_input = np.zeros(shape=(nb_train*9, 1, 40, 40))
    tmp_train_target = np.zeros(shape=(nb_train*9))
    tmp_test_input = np.zeros(shape=(nb_test*9, 1, 40, 40))
    tmp_test_target = np.zeros(shape=(nb_test*9))
    for i in range(9):
        tmp_train_input[i*nb_train:(i+1)*nb_train,:,:,:], tmp_train_target[i*nb_train:(i+1)*nb_train] = take_n_digit(train_input, train_target, i, nb_train)
        tmp_test_input[i*nb_test:(i+1)*nb_test,:,:,:], tmp_test_target[i*nb_test:(i+1)*nb_test] = take_n_digit(test_input, test_target, i, nb_test)   
    return tmp_train_input, tmp_train_target.astype(np.int64), tmp_test_input, tmp_test_target.astype(np.int64)


def load_rotated_digits(path):
    '''
    load previously generated rotated crops of digits of the training video
    '''
    test_target_vid = np.load('rotated_digits/test_target.npy')
    
    names = [nm for nm in os.listdir(path) if '.jpg' in nm]  # make sure to only load .jpg
    names.sort()  # sort file names
    ic_digit = skimage.io.imread_collection([os.path.join(path, nm) for nm in names])
    imgs = skimage.io.concatenate_images(ic_digit)
    
    tmp_imgs = []
    for im in imgs:
        im = rgb2gray(im)
        thresh = threshold_otsu(im)
        im = im > thresh
        tmp_imgs.append(im.astype(int))
        
    test_im=np.array(tmp_imgs).astype(np.float32).reshape(imgs.shape[0],1,40,40)
    return test_im, test_target_vid

def get_data(new_rotations=False):
    '''
    loads train and test data (input + target) and performs processing and random rotations
    '''
    # unzip files
    with zipfile.ZipFile('MNIST40/test.mat.zip', 'r') as zip_ref:
        zip_ref.extractall('MNIST40/')
    with zipfile.ZipFile('MNIST40/training_and_validation.mat.zip', 'r') as zip_ref:
        zip_ref.extractall('MNIST40/')
        
    # load data
    test_input, test_target = load_data('MNIST40/test.mat')
    train_input, train_target = load_data('MNIST40/training_and_validation.mat')

    # for each digit, select randomly 3000 samples for train_input and 500 for test_inputs to have a uniform dataset
    train_input, train_target, test_input, test_target = create_uniform_dataset(train_input, train_target, test_input, test_target, 3000, 500)

    # load rotated digits of training video
    if new_rotations:
        perform_rotation('crop_object/', 'rotated_digits/rot_im', 'rotated_digits/test_target.npy')
    rot_imgs, rot_imgs_target = load_rotated_digits('rotated_digits/')

    # reshape data to remove the 1 channel
    train_input = train_input.reshape(train_input.shape[0], 40, 40)
    test_input = test_input.reshape(test_input.shape[0], 40, 40)

    # generate random rotations of train_input and test_input to make our classifier rotation-invariant
    train_input_rot, train_target_rot = random_rotations(train_input, train_target, train_input.shape[0])
    test_input_rot, test_target_rot = random_rotations(test_input, test_target, test_input.shape[0])

    # concatenate generated rotated data to "normal" data
    train_input = np.vstack((train_input, train_input_rot))
    train_target = np.hstack((train_target, train_target_rot))
    test_input = np.vstack((test_input, test_input_rot))
    test_target = np.hstack((test_target, test_target_rot))

    # reshape again with the 1 channel
    train_input = train_input.reshape(train_input.shape[0], 1, 40, 40).astype(np.float32)
    test_input = test_input.reshape(test_input.shape[0], 1, 40, 40).astype(np.float32)

    # add rotated digits of traininf video to the train set (train_input)
    train_input = np.vstack((train_input, rot_imgs))
    train_target = np.hstack((train_target, rot_imgs_target))

    # shuffle train and test set
    train_input, train_target = shuffle(train_input, train_target, random_state=0)
    test_input, test_target = shuffle(test_input, test_target, random_state=0)

    # convert all from numpy to torch tensors
    train_input = torch.from_numpy(train_input)
    train_target = torch.from_numpy(train_target)
    test_input = torch.from_numpy(test_input)
    test_target = torch.from_numpy(test_target)

    # check sizes before training
    print('Train input size:  ', train_input.size())
    print('Train target size: ', train_target.size())
    print('Test input size:   ', test_input.size())
    print('Test target size:  ', test_target.size())

    return train_input, train_target, test_input, test_target