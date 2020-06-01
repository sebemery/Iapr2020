# Special Project
Equation elements classification by tracking the robot's position.

## Principle
Track the position of a robot in each frame of a 2 FPS input video. Each time a the robot passes through an equation element in the arena, the element 
is sent in a classifier to recognize it and added to the equation. At the end of the sequence display the result of the equation.

## General Information

### Packages needed to run the executables
* numpy
* scipy
* scikit-image
* matplotlib
* scikit-learn
* OpenCV
* Pytorch
* torchvision

### Executable `for the TA's`

#### Main executable
The main.py file runs the code to generate the output video. It first extract and load images from input video in a folder. Then there is a main loop 
iterating over all images. It first get the position of all objects (digits and operators) doing segmentation on the first image. Then overlap test are
performed to check when the robot is on an object. For positive test, the object overlapping the robot is cropper from the first image and sent to a 
classifier. Result of classifier is stored and the whole operation is evalutated at the end of the loop. Finally, all the processed image are stacked together
to make the output video.

Has to be called this way in the command line : python main.py --input <input_path> --output <output_path>. 
by default (python main.py), the code runs with the following file : "../data/robot_parcours_1.avi" 


#### Operator classifier training
The KNN training can be redone and the model saved automaticaly with the right filename and folder location. It is done by augmenting the three operators 
addition, substraction and multiplication by 360 rotations. In the folder operators_classifier run 'main_operator'. To see the result of the gridsearchCV for
this model run 'train_knn_classifier'. All this takes a short time,but it is not necessary for the main to run. 

#### Digit classifier training 
The CNN training can be done again automatically. This training is done by augmenting the affNIST trainset by performing random rotations. In addition, 
crop images of the digits in the video were also augmented with random rotations. In the folder digits_classifier, run the file 'main_digits.py'. 
It will augment both datasets (affNIST and crop images) and then train the CNN classifier over 50 epochs. 



## Data

Name                  |      Content        
----------------------|--------------------------------
robot_parcours_1.avi  |         Video                  
original_operators.png|  Image of all operators 
affNIST dataset       | handwritten digit {0-9} with target

## Folders 

### src 
The src contains all the code to runs the main.py and get the output video, appart for the code needed to run and train the classifiers which are in separated folders.

* main.py : Main file to generate the output video given the input video
* helpers.py : contains the function to extract images from the input video and load the images 
* element_location.py : contains the code needed to binarize and get the bounding boxes of the objects (digits and operators)
* robot_tracking.py : contains the functions to get the robot bounding boxes

### operator_classifier 
The principle of our operator classifier rely on a two stage classifier. First, it recognizes the number of disjoint contours from a binary operator image
with the find_contours function provided in scikit-image. it can recognize the division and equal operators. Then, if it is an addition, substraction 
or multiplication it goes through a KNN classifier train with Fourier descriptors as features.

* Load_image.py : contain a function to load images
* masks_generator.py : contain a function that generate the operator masks use for training
* Training_data.py : contain a function to rotate the digit and a function that build the training data with or without rotations 
* features.py 
	* Fourier_descriptor : a function that compute the DFT of an binary operator's contour and the number of disjoint contour
	* get_features : if set to training compute all the training operator's feature and build the target, otherwise only compute the features of a single binary operator image.
* train_knn_classifier.py 
	* Contain a main to a gridsearchCV on the number of neighbors to use with the default training set (360 rotations)
	* train_knn : train the KNN classifier given features, targets and the number of neighbors to use
* main_operator.py : A main that reproduce our KNN classifier
* operators_classifiers : The actual classifier used in the main


### digits_classifier
To classify the digits encoutered by the robot, a CNN was used with cross entropy as loss criterion and Adam as optimizer. To make it invariant to the possible rotations 
of the digits in the videos we augmented affNIST dataset with random rotations. Moreover, it was also necessary to train the CNN with crop images of the training video augmented 
with random rotations as well (as some particular digit of the video was leading to false classification).

* CNN.py : contains the convolutional neural network
* generate_rotated_imgs.py : contains functions to load crop images of the video, process them and augment them with random rotations
* generate_trainset : contains functions allowing to augment dataset (with rotated affNIST and crop images), process it and to load it in the main
* train_CNN.py : contains functions allowing to train the CNN and compute the number of errors it does
* main_digits : main that generates augmented dataset and trains the CNN on it and saves it in model folder


## Contributors
Canton Diego, Emery Sébastien and Heusghem Pauline
