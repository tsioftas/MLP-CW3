from modulefinder import Module
import data_utils as du

import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, CrossEntropyLoss
from torch.optim import Adam

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
    # Get the data
    data: du.Data = du.get_all_data()
    train_x = data.get['train_x']
    train_y = data.get['train_y']
    dev_x = data.get['dev_in_x'].append(data.get['dev_out_x'])
    dev_y = data.get['dev_in_y'].append(data.get['dev_out_y'])
    # Define the model
    model = SimpleCNN(hidden_layers)
    print(f"Model:\n{model}")
    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=0.07)
    # Define the loss function
    criterion = CrossEntropyLoss()
    # Check if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    # convert to tensors
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

 
if __name__ == "__main__":
    run_cnn(3)

    