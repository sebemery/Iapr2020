from masks_generator import generate_masks
from training_data import load_train_data_knn
from features import get_features
from train_knn_classifier import train_knn

if __name__ == "__main__":
    
    # Generate the mask -> already done just here to show how masks were generated
    generate_masks()
    
    # Load the training data -> data augmentation 360 rotated version of the masks [add,minus,mul]
    operators = load_train_data_knn(360,True)

    # Get features -> four fourier descriptors plus the number of disjoint contours
    features, targets = get_features(operators, 6 , training = True)
    
    # Only take Fourier Descriptors for knn
    features_knn = features[:,:4]
    
    # Train and save the model 
    train_knn(features_knn,targets,5)
