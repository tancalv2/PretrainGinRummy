import numpy as np

################################################# Loss Weights #################################################
def lossWeights(actions, loss_weight=None): 
    '''
    Loss weights for each action/class, default weight is equal class weighting
    '''
    weights = np.ones(actions.shape[1])
    if loss_weight == 'icf' or loss_weight == 'log_icf':
        # adapted from https://github.com/ultralytics/yolov3/issues/249
        weights = np.sum(actions, axis=0).astype(np.int) + 1    # get class weight frequency; pad each weight by 1
        if loss_weight == 'log_icf':
            weights = np.log(weights) + 1
        weights = 1 / weights  # inverse frequency 
        weights /= weights.sum()  # normalize 
    return weights


################################################# Balance #################################################
def balanceClasses(states, actions):
    '''
    Balance the states, actions using the least non-zero class of actions
    '''
    class_count = np.sum(actions, axis=0)
    zero_count_ind = np.where(class_count == 0)[0]
    class_count[zero_count_ind] = 1e10
    min_count = np.min(class_count)
    min_count_ind = np.argmin(class_count)
    
    # Get indices of current class 
    action_ind = np.where(actions[:,min_count_ind] == 1)[0]
    actions_bal = actions[action_ind]
    states_bal = states[action_ind]
    for curr_ind in range(actions.shape[1]): 
        if curr_ind not in zero_count_ind and curr_ind != min_count_ind:
            action_ind = np.where(actions[:,curr_ind] == 1)[0]
            actions_temp = actions[action_ind]
            states_temp = states[action_ind]

            random_indices = np.random.choice(actions_temp.shape[0], size=min_count, replace=False)
            actions_temp = actions_temp[random_indices]
            states_temp = states_temp[random_indices]

            actions_bal = np.concatenate((actions_bal, actions_temp))
            states_bal = np.concatenate((states_bal, states_temp))
    return states_bal, actions_bal


################################################# Prune States #################################################
def pruneStates(states, stateList=[]):
    ''' States '''
    # (0) ignore current hand
    # states[:,(260-52*5):(260-52*4)] = 0
    # (1) ignore top card feature
    # states[:,(260-52*4):(260-52*3)] = 0
    # (2) ignore dead cards feature
    # states[:,(260-52*3):(260-52*2)] = 0
    # (3) ignore opponent known cards feature
    # states[:,(260-52*2):(260-52*1)] = 0
    # (4) ignore unknown cards feature
    # states[:,(260-52):(260-52*0)] = 0
    # prunable states
    prune_states = {'currHand': 0, 'topCard': 1,
                    'deadCard': 2, 'oppCard': 3,
                    'unknownCard': 4}    
    for s in stateList:
        try: 
            print('pruning state: {}'.format(s))
            states[:,(260-52*(5-prune_states[s])):(260-52*(4-prune_states[s]-1))] = 0
        except:
            print('{} is not a state'.format(s))
            pass
    return states

################################################# Choose Actions #################################################
# chooseable actions
choose_actions = {'all': 0,'draw_pickup': 1,'discard': 2,'knock': 3,'knock_bin': 4}
def chooseActions(actions, classes, actionChoice):
    ''' Actions '''
    # (0) all:          actions
    #       (x) score_player: actions[:,0:2]
    # (1) draw:         actions[:,2:4]
    #       (x) deadHand:     actions[:,4]
    #       (x) gin:          actions[:,5]
    # (2) discard:      actions[:,6:58]
    # (3) knock:        actions[:,58:]
    # (4) knock_bin:    actions

    if actionChoice == 'all':
        return actions, classes
    elif actionChoice == 'draw':
        return actions[:,2:4], classes[2:4]
    elif actionChoice == 'discard':
        return actions[:,6:58], classes[6:58]
    elif actionChoice == 'knock':
        return actions[:,58:], classes[58:]
    elif actionChoice == 'knock_bin':
        return actions, ["No Knock", "Knock"]

    else:
        print('action selected not allowed')
        return actions, classes