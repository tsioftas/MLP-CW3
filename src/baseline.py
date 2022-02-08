from modulefinder import Module
import data_utils as du

import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear

class SimpleCNN(Module):

    def __init__(self, layers):
        super(SimpleCNN, self).__init__()
        self.input_dim = 127
        self.hidden_dim = 70
        self.output_dim = 1
        seq = []
        for i in range(layers):
            seq.append(Conv2d(self.input_dim, self.hidden_dim))
            seq.append(BatchNorm2d(self.hidden_dim))
            seq.append(ReLU(inplace=True))
            seq.append(MaxPool2d())

        self.cnn_layers = Sequential(
            seq
        )

        self.linear_layers = Sequential(
            Linear(self.hidden_dim, self.output_dim)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def run_cnn(hidden_layers):
    data: du.Data = du.get_all_data()
    train_x = data.get['train']
    train_y = train_x['fact_temperature']
    train_x = train_x[train_x.columns.drop(list(train_x.filter(regex='fact_*')))]
    train_x.drop('climate', inplace=True)
    model = SimpleCNN(hidden_layers)

    # convert to tensors
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

 
if __name__ == "__main__":
    run_cnn(3)

    