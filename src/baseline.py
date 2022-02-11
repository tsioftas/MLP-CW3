import data_utils as du

import numpy as np
import os
from matplotlib import pyplot as plt
import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, MSELoss, Module
from torch.optim import Adam
from torch.autograd import Variable
from collections import OrderedDict

class SimpleCNN(Module):

    def __init__(self, layers):
        super(SimpleCNN, self).__init__()
        self.input_dim = 1
        self.hidden_dim = 10
        self.output_dim = 1
        seq = OrderedDict()
        for i in range(layers):
            if i==0:
                seq[f"conv_{i}"] = Conv2d(self.input_dim, self.hidden_dim, kernel_size=5, dtype=np.float32, padding=3)
            else:
                seq[f"conv_{i}"] = Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=5, dtype=np.float32, padding=3)
            #seq[f"batchnorm_{i}"]  = BatchNorm2d(self.hidden_dim, dtype=np.float32)
            seq[f"relu_{i}"] = ReLU(inplace=True)
            #seq[f"maxpool_{i}"] = MaxPool2d(kernel_size=2, stride=2)

        self.cnn_layers = Sequential(
            seq
        )

        self.linear_layers = Sequential(
            Linear(self.hidden_dim*300, self.output_dim, dtype=np.float32)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def train(epoch: int, model: SimpleCNN,  data: du.Data, criterion,
            train_x, train_y, dev_x, dev_y):
    model.train()
    
    
    # getting the training set
    x_train, y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    x_val, y_val = Variable(dev_x), Variable(dev_y) # change variable name dev -> val
    # converting the data into GPU format
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=0.00001)
    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    output_train = model(x_train).squeeze()
    output_val = model(x_val).squeeze()

    # computing the training and validation loss
    loss_train = criterion(output_train, y_train)
    loss_val = criterion(output_val, y_val)
    train_losses.append(loss_train.detach().numpy())
    val_losses.append(loss_val.detach().numpy())

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch%2 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)

def make_cnn(hidden_layers, E):
    # Get the data
    print("Getting data...")
    data: du.Data = du.get_all_data()
    # Define the model
    print("Defining model...")
    model = SimpleCNN(hidden_layers)
    # Define the loss function
    criterion = MSELoss()
    print(f"Model:\n{model}")
    # Check if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    # extract data
    train_x = data.get['shifts_canonical_train_x']
    train_y = data.get['shifts_canonical_train_y']
    dev_x = data.get['shifts_canonical_dev_in_x'].append(data.get['shifts_canonical_dev_out_x'])
    dev_y = data.get['shifts_canonical_dev_in_y'].append(data.get['shifts_canonical_dev_out_y'])
    # convert data to tensors
    # Add dummy variables (padding for convolution)
    for i in range(3):
        dev_x.insert(len(dev_x.columns), f"Dummy_var_{i}", [0 for _ in range(dev_x.shape[0])])
        train_x.insert(len(train_x.columns), f"Dummy_var_{i}", [0 for _ in range(train_x.shape[0])])
    N_SAMPLES, D = train_x.shape
    D1 = 14
    D2 = 9
    assert D1*D2 == D, f"Have {D} features, cannot organize them in {D1} rows and {D2} columns!"
    train_x = torch.from_numpy(train_x.to_numpy().reshape((N_SAMPLES, D1, D2))).unsqueeze(1)
    train_y = torch.from_numpy(train_y.to_numpy())
    N_SAMPLES, D = dev_x.shape
    print(f"Shape: {dev_x.shape}")
    dev_x = torch.from_numpy(dev_x.to_numpy().reshape((N_SAMPLES, D1, D2))).unsqueeze(1)
    dev_y = torch.from_numpy(dev_y.to_numpy())
    epochs = [i for i in range(E)]
    print(f"Strarted training for {E} epochs...")
    for epoch in epochs:
        train(epoch, model, data, criterion, train_x, train_y, dev_x, dev_y)  
    return model  

 
if __name__ == "__main__":
    train_losses = []
    val_losses = []
    E=30
    epochs = [i for i in range(E)]
    m = make_cnn(3, E)
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    c = 0
    while os.path.isfile(f"plots/plot_{c}.svg"):
        c += 1
    plt.savefig(f"plot_{c}.svg")
    plt.show()

    