from generate_rotated_data import get_data
from train_CNN import train_cnn

# generate data + rotated data to train the classifier
train_input, train_target, test_input, test_target = get_data()

# train model + save it in model folder of project 
train_cnn(train_input, train_target, test_input, test_target)
