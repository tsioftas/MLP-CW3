import data_utils as du
from constants import BATCH_SIZE

import math
import numpy as np
import os
from matplotlib import pyplot as plt
import torch
from torch.nn import Sequential, Conv1d, BatchNorm2d, ReLU, MaxPool2d, Linear, MSELoss, Module
from torch.optim import Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from collections import OrderedDict

class SimpleCNN(Module):

    def __init__(self, layers, D_in):
        super(SimpleCNN, self).__init__()
        self.output_dim = 1

        self.in_channels_1st = D_in
        self.in_channels_hidden = 512
        self.out_channels_1st = self.in_channels_hidden
        self.out_channels_hidden = 512
        seq = OrderedDict()
        for i in range(layers):
            if i==0:
                seq[f"conv_{i}"] = Conv1d(self.in_channels_1st, self.out_channels_1st, kernel_size=5, dtype=torch.float, padding=2, bias=False)
            else:
                seq[f"conv_{i}"] = Conv1d(self.in_channels_hidden, self.out_channels_hidden, kernel_size=5, dtype=torch.float, padding=2, bias=True)
            #seq[f"batchnorm_{i}"]  = BatchNorm2d(self.hidden_dim, dtype=np.float32)
            seq[f"relu_{i}"] = ReLU(inplace=True)
            #seq[f"maxpool_{i}"] = MaxPool2d(kernel_size=2, stride=2)

        self.cnn_layers = Sequential(
            seq
        )

        self.linear_layers = Sequential(
            Linear(self.out_channels_hidden, self.output_dim, dtype=torch.float)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class Dataset:
        def __init__(self, features, targets):
            self.features = features
            self.targets = targets

        def __len__(self):
            return (self.features.shape[0])

        def __getitem__(self, idx):
            dct = {
                # TODO: Assuming dimensionality of data for 2d convolutional layer 
                'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
                'y' : torch.tensor(self.targets[idx], dtype=torch.float)            
            }
            return dct

def train(epoch: int, model: SimpleCNN,  data: du.Data, criterion,
            train_dataloader: DataLoader, valid_dataloader: DataLoader):
    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=0.00001)

    # Training data
    model.train()
    train_loss_avg = 0
    for train_data in train_dataloader:
        # getting the training set
        x_train, y_train = Variable(train_data['x']), Variable(train_data['y'])
        # converting the data into GPU format
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
        # prediction for training and validation set
        output_train = model(x_train).squeeze()
        if output_train.shape != y_train.shape:
            print("OPPAS")
        train_loss = criterion(output_train, y_train)
        train_loss.backward()
        optimizer.step()

        train_loss_avg += train_loss.item()
    train_loss_avg /= len(train_dataloader)
    train_losses.append(train_loss_avg)
    
    # Validation data
    model.eval()
    valid_loss_avg = 0
    for valid_data in valid_dataloader:
        # getting the validation set
        x_val, y_val = Variable(valid_data['x']), Variable(valid_data['y']) # change variable name dev -> val
        if torch.cuda.is_available():
            x_val = x_val.cuda()
            y_val = y_val.cuda()
        output_val = model(x_val).squeeze()
        if output_val.shape != y_val.shape:
            print("OPPAS_2")
        valid_loss = criterion(output_val, y_val)
        valid_loss_avg += valid_loss.item()
    valid_loss_avg /= len(valid_dataloader)
    val_losses.append(valid_loss_avg)

    if epoch%2 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :', valid_loss_avg)

def make_cnn(hidden_layers, E):
    # Get the data
    print("Getting data...")
    data: du.Data = du.get_all_data()
    # extract data
    train_x = data.get['shifts_canonical_eval_in_x']
    train_y = data.get['shifts_canonical_eval_in_y']
    val_x = data.get['shifts_canonical_dev_in_x'].append(data.get['shifts_canonical_dev_out_x'])
    val_y = data.get['shifts_canonical_dev_in_y'].append(data.get['shifts_canonical_dev_out_y'])
    N_SAMPLES_TRAIN, D = train_x.shape
    # convert from pandas to numpy
    train_x = train_x.to_numpy().reshape(N_SAMPLES_TRAIN, D, 1)
    train_y = train_y.to_numpy()
    print(f"Training data features shape: {train_x.shape}")
    N_SAMPLES_VALID, _ = val_x.shape
    val_x = val_x.to_numpy().reshape(N_SAMPLES_VALID, D, 1)
    val_y = val_y.to_numpy()
    print(f"Validation data features shape: {val_x.shape}")
    # numpy to pytorch
    train_dataset = Dataset(train_x, train_y)
    valid_dataset = Dataset(val_x, val_y)
    train_data_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, BATCH_SIZE, shuffle=False)
    
    # Define model
    print("Defining model...")
    model = SimpleCNN(hidden_layers, D)
    # and the loss function
    criterion = MSELoss()
    print(f"Model:\n{model}")
    # Check if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    
    # Start training
    epochs = [i for i in range(E)]
    print(f"Strarted training for {E} epochs...")
    for epoch in epochs:
            train(epoch, model, data, criterion, train_data_loader, valid_data_loader)  
    return model  

 
if __name__ == "__main__":
    train_losses = []
    val_losses = []
    E=3
    epochs = [i for i in range(E)]
    m = make_cnn(3, E)
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, val_losses, label="val_loss")
    c = 0
    while os.path.isfile(f"plots/plot_{c}.svg"):
        c += 1
    plt.savefig(f"plot_{c}.svg")
    plt.show()

    