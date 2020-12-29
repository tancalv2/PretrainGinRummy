import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(input_size, input_size*2)
        self.l2 = nn.Linear(input_size*2, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, features):
        x = self.l1(features)
        x = self.sig(x)
        x = self.l2(x)
        return self.sig(x)
