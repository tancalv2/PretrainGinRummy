import copy
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from MLP import MLP
from ginDataset import ginDataset
from data_process import *
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
def load_train_data(data_pth, plot_pth, numGames, batch_size, state, action, pruneStatesList=[], actionChoice='all', balance=False, visualize=False):
    '''
    Load train data
    '''
    if action in state_action_pair[state]:
        print('loading {} games dataset from {}'.format(numGames, data_pth))
        states = np.load('{}/s_{}k.npy'.format(data_pth, numGames//1000))
        actions = np.load('{}/a_{}k.npy'.format(data_pth, numGames//1000))

        # prune states
        states = pruneStates(states, pruneStatesList)

        # choosable actions
        actions, classes = chooseActions(actions, all_classes, actionChoice)

        # balance classes
        if balance:
            states, actions = balanceClasses(states, actions)

        # Visualize action classes distribution
        if visualize:
            visualizeClasses(plot_pth, actions, classes)

        # split train/val
        data_train, data_val, label_train, label_val = train_test_split(states, actions, test_size=0.3, random_state=421)

        train_loader = load_data(data_train, label_train, batch_size, shuffle=True)
        val_loader = load_data(data_val, label_val, batch_size, shuffle=False)

        return train_loader, val_loader, classes
    else:
        print('illegeal state-action pair')
        return _, _, _
    


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
def load_model(lr=0.001, input_size=None, output_size=None, model=None, pre_train=False, model_PT=None, device='cpu'):
    loss_fnc = torch.nn.MSELoss()
    # if model is None:
    model = MLP(input_size, output_size).to(device)
    if pre_train:
        pre_train_model = torch.load(model_PT, map_location=device)
        model.l1.weight = pre_train_model.l1.weight.to(device)
        model.l1.bias = pre_train_model.l1.bias.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return model, loss_fnc, optimizer
    # return model, loss_fnc, _


################################################# Evaluate #################################################
def evaluate(model, data_loader, loss_fnc, device='cpu'):

    total_corr = 0
    accum_loss = 0
    for i, data in enumerate(data_loader):
        inputs, labels = data
        inputs = inputs.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor).to(device)

        outputs = model(inputs)
        batch_loss = loss_fnc(input=outputs, target=labels)

        target = torch.argmax(labels, axis=1)
        pred = torch.argmax(outputs, dim=1)
        corr = pred == target
        total_corr += int(corr.sum())
        accum_loss += batch_loss

    acc = float(total_corr)/len(data_loader.dataset)
    loss = accum_loss/(i+1)
    return acc, loss.item() 


################################################# Train #################################################
def train(train_loader, val_loader, plot_pth, batch_size=1000, lr=0.001, epochs=100, verbose=False, pre_train=False, model_PT=None, device='cpu'):

    input_size = len(train_loader.dataset.features[0])
    output_size = len(val_loader.dataset.labels[0])

    model, loss_fnc, optimizer = load_model(lr, input_size, output_size,
                                            pre_train=pre_train, model_PT=model_PT, device=device)
    
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
            batch_loss = loss_fnc(input=outputs, target=labels)
            accum_loss += batch_loss
            batch_loss.backward()
            optimizer.step()

            target = torch.argmax(labels, axis=1)
            pred = torch.argmax(outputs, dim=1)
            corr = pred == target
            total_corr += int(corr.sum())

        # evaluate per epoch
        vacc, vloss = evaluate(model, val_loader, loss_fnc, device)
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