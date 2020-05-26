import torch
import torchvision
import torch.optim as optim
from CNN import ConvNet

def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    '''
    given model, data input and data target, computes the number of errors between the prediction using model and the target
    '''
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size)) 
        _, predicted_classes = torch.max(output, 1) 
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]: # if the prediction is not right, increase number of errors
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors

def train_cnn(train_input, train_target, test_input, test_target): 
    '''
    given a model and a training + target set, trains the model over 100 epochs with a mini-batch size of 1000
    computes and prints accuracy
    saves trained model in folder model of project
    '''
    model = ConvNet()
    mini_batch_size = 1000
    criterion = torch.nn.CrossEntropyLoss()  # Loss criterion = Cross Entropy
    optimizer = optim.SGD(model.parameters(), lr = 1e-1) # Optimizer = SGD
    nb_epochs = 100

    print('\n')
    print('>>> Model training ...')
    print('\n')
    
    for e in range(nb_epochs):
        losses = 0
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size)) # prediction
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size)) # computation of the loss
            model.zero_grad() # set gradient to zero
            loss.backward() # backward pass 
            optimizer.step() # optimizaiton of the weights
            losses += loss.item()
            
        print('Iteration {}:    loss = {:0.2f}'.format(e, losses))

    # Computing number of errors on the test set and train set
    nb_train_errors = compute_nb_errors(model, train_input, train_target, mini_batch_size)
    nb_test_errors = compute_nb_errors(model, test_input, test_target, mini_batch_size)

    # Train and test accuracy
    print('\n')
    print('Train accuracy:   {:0.2f}%'.format(100 - ((100 * nb_train_errors) / train_input.size(0))))
    print('Test accuracy:    {:0.2f}%'.format(100 - ((100 * nb_test_errors) / test_input.size(0))))

    # save model
    torch.save(model.state_dict(), './model/digits_classifier')

