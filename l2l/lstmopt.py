import torch
from torch.nn import functional as F
import torchvision as tv
import learn2learn as l2l
import tqdm
import torch.nn as nn

def accuracy(predictions, targets):
    """Returns mean accuracy over a mini-batch"""
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        inp_size = 28*28
        layer_size = 20
        # todo these should use sigmoid activations (currently just fully connected)
        self.fc1 = nn.Linear(inp_size, layer_size)
        self.fc_final = nn.Linear(layer_size, 10)
        
#         self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = torch.nn.Dropout2d(0.25)
#         self.dropout2 = torch.nn.Dropout2d(0.5)
#         self.fc1 = torch.nn.Linear(9216, 128)
#         self.fc2 = torch.nn.Linear(128, 10)

    def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
        print(x.shape)
        x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
        x = self.fc_final(x)
        output = F.log_softmax(x, dim=1)
        # the MNISTNet returns loss here??
        return output

class HypergradTransform(torch.nn.Module):
    """Hypergradient-style per-parameter learning rates"""

    def __init__(self, param, lr=0.01):
        super(HypergradTransform, self).__init__()
        self.lr = lr * torch.ones_like(param, requires_grad=True)
        self.lr = torch.nn.Parameter(self.lr)

    def forward(self, grad):
        return self.lr * grad
    

class LSTMOptimizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 20
        self.recurs = nn.LSTMCell(1, self.hidden_size)
        self.recurs2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.output = nn.Linear(self.hidden_size, 1)
        
    def forward(self, inp, hidden, cell):
        hidden0, cell0 = self.recurs(inp, (hidden[0], cell[0]))
        hidden1, cell1 = self.recurs2(hidden0, (hidden[1], cell[1]))
        return self.output(hidden1), (hidden0, hidden1), (cell0, cell1)
    
    