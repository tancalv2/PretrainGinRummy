import torch.nn as nn

class MLP_base(nn.Module):
    '''
    1 Hidden Layer: size input_size*2
    '''
    def __init__(self, input_size, output_size, activation='sig'):
        super(MLP_base, self).__init__()
        self.l1 = nn.Linear(input_size, input_size*2)
        self.l2 = nn.Linear(input_size*2, output_size)
        if activation == 'relu':
            self.act_fnc = nn.ReLU()
        elif activation == 'tanh':
            self.act_fnc = nn.Tanh()
        else:
            self.act_fnc = nn.Sigmoid()
        self.sfx = nn.Softmax(dim=1)

    def forward(self, features):
        x = self.l1(features)
        x = self.act_fnc(x)
        x = self.l2(x)
        return self.sfx(x)
