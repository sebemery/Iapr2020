import torch
import scipy.io as sio
import numpy as np
import random
from skimage.filters import threshold_otsu
from skimage.transform import rescale
from skimage.transform import rotate
import zipfile

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

def choose_n_random(nb_samples, data, data_target):
    '''
    choose nb_samples random samples among data and data_target
    '''
    random_ind = random.sample(range(1, len(data_target)), nb_samples)
    data = data[random_ind,:,:,:]
    data_target = data_target[random_ind]
    return data, data_target

def remove9(data, data_target):
    '''
    remove 9 digit since we now there won't be any in the videos 
    '''
    ind = []
    for i in range(len(data_target)):
        if(data_target[i]!=9):
            ind.append(i)
    new_data = data[ind]
    new_data_target = data_target[ind]
    return new_data, new_data_target

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

def get_data():
    '''
    loads train and test data (input + target) and performs processing and random rotations
    '''
    # unzip files
    with zipfile.ZipFile('MNIST40/test.mat.zip', 'r') as zip_ref:
        zip_ref.extractall('MNIST40/')
    with zipfile.ZipFile('MNIST40/training_and_validation.mat.zip', 'r') as zip_ref:
        zip_ref.extractall('MNIST40/')
        
    # load data
    test_size = 8000
    train_size = 30000
    test_input, test_target = load_data('MNIST40/test.mat')
    train_input, train_target = load_data('MNIST40/training_and_validation.mat')

    # remove 9 from dataset since there will only be digits in [0,8]
    train_input, train_target = remove9(train_input, train_target)
    test_input, test_target = remove9(test_input, test_target)

    # choose a at random test and train data of size test_size and train_size 
    test_input, test_target = choose_n_random(test_size, test_input, test_target)
    train_input, train_target = choose_n_random(train_size, train_input, train_target)

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