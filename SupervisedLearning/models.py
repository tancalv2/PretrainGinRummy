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

class MLP_2HL(nn.Module):
    '''
    2 Hidden Layers: size input_size*2, input_size*2
    '''
    def __init__(self, input_size, output_size, activation='sig'):
        super(MLP_2HL, self).__init__()
        self.l1 = nn.Linear(input_size, input_size*2)
        self.l2 = nn.Linear(input_size*2, input_size*2)
        self.l3 = nn.Linear(input_size*2, output_size)
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
        x = self.act_fnc(x)
        x = self.l3(x)
        return self.sfx(x)

class MLP_3HL(nn.Module):
    '''
    3 Hidden Layers: size input_size*2, input_size*2, input_size
    '''
    def __init__(self, input_size, output_size, activation='sig'):
        super(MLP_3HL, self).__init__()
        self.l1 = nn.Linear(input_size, input_size*2)
        self.l2 = nn.Linear(input_size*2, input_size*2)
        self.l3 = nn.Linear(input_size*2, input_size)
        self.l4 = nn.Linear(input_size, output_size)
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
        x = self.act_fnc(x)
        x = self.l3(x)
        x = self.act_fnc(x)
        x = self.l4(x)
        return self.sfx(x)

class MLP_4HL(nn.Module):
    '''
    4 Hidden Layers: size input_size*2, input_size*4, input_size*4, , input_size*2
    '''
    def __init__(self, input_size, output_size, activation='sig'):
        super(MLP_4HL, self).__init__()
        self.l1 = nn.Linear(input_size, input_size*2)
        self.l2 = nn.Linear(input_size*2, input_size*4)
        self.l3 = nn.Linear(input_size*4, input_size*4)
        self.l4 = nn.Linear(input_size*4, input_size*2)
        self.l5 = nn.Linear(input_size*2, output_size)
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
        x = self.act_fnc(x)
        x = self.l3(x)
        x = self.act_fnc(x)
        x = self.l4(x)
        x = self.act_fnc(x)
        x = self.l5(x)
        return self.sfx(x)

class MLP_4HL2(nn.Module):
    '''
    4 Hidden Layers: size input_size*2, input_size*2, input_size*2, , input_size*2
    '''
    def __init__(self, input_size, output_size, activation='sig'):
        super(MLP_4HL2, self).__init__()
        self.l1 = nn.Linear(input_size, input_size*2)
        self.l2 = nn.Linear(input_size*2, input_size*2)
        self.l3 = nn.Linear(input_size*2, input_size*2)
        self.l4 = nn.Linear(input_size*2, input_size*2)
        self.l5 = nn.Linear(input_size*2, output_size)
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
        x = self.act_fnc(x)
        x = self.l3(x)
        x = self.act_fnc(x)
        x = self.l4(x)
        x = self.act_fnc(x)
        x = self.l5(x)
        return self.sfx(x)

class MLP_2HL_wide(nn.Module):
    '''
    2 Hidden Layers: size input_size*2, input_size*2
    '''
    def __init__(self, input_size, output_size, activation='sig'):
        super(MLP_2HL_wide, self).__init__()
        self.l1 = nn.Linear(input_size, input_size*4)
        self.l2 = nn.Linear(input_size*4, input_size*4)
        self.l3 = nn.Linear(input_size*4, output_size)
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
        x = self.act_fnc(x)
        x = self.l3(x)
        return self.sfx(x)