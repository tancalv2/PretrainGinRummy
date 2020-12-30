import copy
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models import *
from MLP import *
from ginDataset import ginDataset
from data_preprocess import *
from visualize_data import *
from utils import state_action_pair, all_classes

################################################# Load Data #################################################
def load_data(data, label, batch_size=1000, shuffle=False):
    '''
    Load dataset according to batch_size given
    '''
    data_set = ginDataset(data, label)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader


################################################# Load Test Data #################################################
def load_train_data(data_pth, plot_pth, numGames, batch_size, pruneStatesList=[], actionChoice='all', balance=False, loss_weight=None, multi_data_pth={}, visualize=False):
    '''
    Load train data
    '''
    if data_pth is not None:
        print('loading {} games dataset from "{}"'.format(numGames, data_pth))
        states = np.load('{}/s_{}k.npy'.format(data_pth, numGames//1000))
        actions = np.load('{}/a_{}k.npy'.format(data_pth, numGames//1000))

        # check for additional data paths
        for key, data_pth2 in multi_data_pth.items():
            if os.path.exists(data_pth2):
                print('concatenating {} dataset from "{}"'.format(key, data_pth2))
                states2 = np.load('{}/s_{}k.npy'.format(data_pth2, numGames//1000))
                states = np.concatenate((states, states2))
                del states2
                actions2 = np.load('{}/a_{}k.npy'.format(data_pth2, numGames//1000))
                actions = np.concatenate((actions, actions2))
                del actions2
            else:
                print('Path "{}" does not exist'.format(data_pth2))

        # prune states
        states = pruneStates(states, pruneStatesList)

        # choosable actions
        actions, classes = chooseActions(actions, all_classes, actionChoice)

        # balance classes
        if balance:
            states, actions = balanceClasses(states, actions)
        
        # obtain loss weights for each class/action
        weights = lossWeights(actions, loss_weight)
        
        # split train/val
        data_train, data_val, label_train, label_val = train_test_split(states, actions, test_size=0.3, random_state=421)

        # Visualize action classes distribution for all, train, and validation splits
        if visualize:
            visualizeClasses(plot_pth, actions, classes, ['All', 'b', 'all'])
            visualizeClasses(plot_pth, label_train, classes, ['Train Split', 'g', 'train'])
            visualizeClasses(plot_pth, label_val, classes, ['Validation Split', 'r', 'val'])

        train_loader = load_data(data_train, label_train, batch_size, shuffle=True)
        val_loader = load_data(data_val, label_val, batch_size, shuffle=False)

        return train_loader, val_loader, torch.Tensor(weights), classes
    else:
        print('incorrect data path')
        return None, None, None, None
    


################################################# Load Test Data #################################################
def load_test_data(data_pth, numGames, state, action, pruneStatesList, actionChoice):
    '''
    Load test data
    '''
    if action in state_action_pair[state]:
        print('loading {} games dataset from {}'.format(numGames, data_pth))
        states_test = np.load('{}/s_{}k.npy'.format(data_pth, numGames//1000))
        actions_test = np.load('{}/a_{}k.npy'.format(data_pth, numGames//1000))

        # prune states
        states_test = pruneStates(states_test, pruneStatesList)

        # choosable actions
        actions_test, classes = chooseActions(actions_test, all_classes, actionChoice)
        test_loader = load_data(states_test, actions_test)
        return test_loader, classes
    else:
        print('illegeal state-action pair')
        return _, _


################################################# Load Model #################################################
def load_model(lr=0.001, input_size=None, output_size=None, model_fnc='MLP_base', activation='sig', loss='MSE', weights=None, pre_train=False, model_PT=None, device='cpu'):

    if model_fnc == 'MLP_2HL':
        model = MLP_2HL(input_size, output_size, activation).to(device)
    else:
        model = MLP_base(input_size, output_size, activation).to(device)

    if loss == 'CELoss':
        loss_fnc = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    else:
        loss_fnc = torch.nn.MSELoss()
    
    if pre_train:
        pre_train_model = torch.load(model_PT, map_location=device)
        model.l1.weight = pre_train_model.l1.weight.to(device)
        model.l1.bias = pre_train_model.l1.bias.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, loss_fnc, optimizer


################################################# Evaluate #################################################
def evaluate(model, data_loader, loss_fnc=torch.nn.MSELoss(), loss='MSE', device='cpu'):

    total_corr = 0
    accum_loss = 0
    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs = inputs.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor).to(device)

        outputs = model(inputs)
        target = torch.argmax(labels, axis=1)
        pred = torch.argmax(outputs, dim=1)
        if loss == 'CELoss':
            batch_loss = loss_fnc(input=outputs, target=target)
        else:
            batch_loss = loss_fnc(input=outputs, target=labels)
        accum_loss += batch_loss

        corr = pred == target
        total_corr += int(corr.sum())

    acc = float(total_corr)/len(data_loader.dataset)
    loss = accum_loss/(i+1)
    return acc, loss.item() 


################################################# Train #################################################
def train(train_loader, val_loader, plot_pth, batch_size=1000, lr=0.001, epochs=100, verbose=False, model_fnc='MLP_base', activation='sig', loss='MSE', weights=None, pre_train=False, model_PT=None, device='cpu'):

    input_size = len(train_loader.dataset.features[0])
    output_size = len(val_loader.dataset.labels[0])

    model, loss_fnc, optimizer = load_model(lr, input_size, output_size, model_fnc=model_fnc, activation=activation,
        loss=loss, weights=weights, pre_train=pre_train, model_PT=model_PT, device=device)
    
    max_val_acc = 0
    min_val_loss = np.inf
    train_acc, train_loss = [], []
    val_acc, val_loss = [], []
    start_time = time.time()

    for epoch in range(epochs):
        accum_loss = 0
        total_corr = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            target = torch.argmax(labels, axis=1)
            pred = torch.argmax(outputs, axis=1)
            if loss == 'CELoss':
                batch_loss = loss_fnc(input=outputs, target=target)
            else:
                batch_loss = loss_fnc(input=outputs, target=labels)
            accum_loss += batch_loss
            batch_loss.backward()
            optimizer.step()

            corr = pred == target
            total_corr += int(corr.sum())

        # evaluate per epoch
        vacc, vloss = evaluate(model, val_loader, loss_fnc, loss, device)
        val_acc.append(vacc)
        val_loss.append(vloss)
        train_loss.append(accum_loss.item()/(i+1))
        train_acc.append(float(total_corr)/len(train_loader.dataset))
        # best acc model
        if vacc > max_val_acc:
            max_val_acc = vacc
            epoch_acc = epoch
            model_acc = copy.deepcopy(model) 
        # best loss model
        if vloss < min_val_loss:
            min_val_loss = vloss
            epoch_loss = epoch
            model_loss = copy.deepcopy(model)

        if verbose:
            # print records
            print("Epoch: {} | Train Loss: {:.8f} | Train acc: {:.6f}"
                .format(epoch + 1, train_loss[epoch], train_acc[epoch]))
            print("              Val Loss: {:.8f} |   Val acc: {:.6f}"
                .format(val_loss[epoch], val_acc[epoch]))
        accum_loss = 0.0
        total_corr = 0


    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total training time elapsed: {:.2f} seconds".format(elapsed_time))
    plotTrain(plot_pth, train_acc, val_acc, epoch_acc, 'Accuracy', batch_size, lr)
    plotTrain(plot_pth, train_loss, val_loss, epoch_loss, 'Loss', batch_size, lr)

    return model, model_acc, model_loss