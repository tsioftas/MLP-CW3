import src.data_utils as du

import torch
from torch.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Linear, CrossEntropyLoss, Module
from torch.optim import Adam
from collections import OrderedDict

class SimpleCNN(Module):

    def __init__(self, layers):
        super(SimpleCNN, self).__init__()
        self.input_dim = 127
        self.hidden_dim = 70
        self.output_dim = 1
        seq = OrderedDict()
        for i in range(layers):
            if i==0:
                seq[f"conv_{i}"] = Conv2d(self.input_dim, self.hidden_dim, kernel_size=3, stride=1, padding = 1)
            else:
                seq[f"conv_{i}"] = Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding = 1)
            seq[f"batchnorm_{i}"]  = BatchNorm2d(self.hidden_dim)
            seq[f"relu_{i}"] = ReLU(inplace=True)
            seq[f"maxpool_{i}"] = MaxPool2d(kernel_size=2, stride=2)

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
    train_x = data.get['shifts_canonical_train_x']
    train_y = data.get['shifts_canonical_train_y']
    dev_x = data.get['shifts_canonical_dev_in_x'].append(data.get['shifts_canonical_dev_out_x'])
    dev_y = data.get['shifts_canonical_dev_in_y'].append(data.get['shifts_canonical_dev_out_y'])
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

    