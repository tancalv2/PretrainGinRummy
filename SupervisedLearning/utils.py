import os

# Available states-action pairs:
state_action_pair = {'all': 'all', # all actions
                    'bpbd': 'draw', # actions 2/3 
                    'apbd': ['discard', 'knock'], # actions 6-57, 58-109
                    'apad': 'knock_bin'} # binary action

# All Possible Classes
all_classes = ['SP0','SP1','Draw','Pickup','DH','GIN',
               'AS_D', '2S_D', '3S_D', '4S_D', '5S_D', '6S_D', '7S_D', '8S_D', '9S_D', 'TS_D', 'JS_D', 'QS_D', 'KS_D',
               'AH_D', '2H_D', '3H_D', '4H_D', '5H_D', '6H_D', '7H_D', '8H_D', '9H_D', 'TH_D', 'JH_D', 'QH_D', 'KH_D',
               'AD_D', '2D_D', '3D_D', '4D_D', '5D_D', '6D_D', '7D_D', '8D_D', '9D_D', 'TD_D', 'JD_D', 'QD_D', 'KD_D',
               'AC_D', '2C_D', '3C_D', '4C_D', '5C_D', '6C_D', '7C_D', '8C_D', '9C_D', 'TC_D', 'JC_D', 'QC_D', 'KC_D',
               'AS_K', '2S_K', '3S_K', '4S_K', '5S_K', '6S_K', '7S_K', '8S_K', '9S_K', 'TS_K', 'JS_K', 'QS_K', 'KS_K',
               'AH_K', '2H_K', '3H_K', '4H_K', '5H_K', '6H_K', '7H_K', '8H_K', '9H_K', 'TH_K', 'JH_K', 'QH_K', 'KH_K',
               'AD_K', '2D_K', '3D_K', '4D_K', '5D_K', '6D_K', '7D_K', '8D_K', '9D_K', 'TD_K', 'JD_K', 'QD_K', 'KD_K',
               'AC_K', '2C_K', '3C_K', '4C_K', '5C_K', '6C_K', '7C_K', '8C_K', '9C_K', 'TC_K', 'JC_K', 'QC_K', 'KC_K']


def create_dir(pth, state, action, model_name):
	# create model and plot directories if do not exist

	# model directories
	state_pth = '{}/models/{}'.format(pth,state)
	if not os.path.exists(state_pth):
	    os.mkdir(state_pth)
	action_pth = '{}/{}'.format(state_pth,action)
	if not os.path.exists(action_pth):
	    os.mkdir(action_pth)
	model_pth = '{}/{}'.format(action_pth,model_name)
	if not os.path.exists(model_pth):
	    os.mkdir(model_pth)

	# plot directories
	state_pth = '{}/plots/{}'.format(pth,state)
	if not os.path.exists(state_pth):
	    os.mkdir(state_pth)
	action_pth = '{}/{}'.format(state_pth,action)
	if not os.path.exists(action_pth):
	    os.mkdir(action_pth)
	plot_pth = '{}/{}'.format(action_pth,model_name)
	if not os.path.exists(plot_pth):
	    os.mkdir(plot_pth)

	# data directory
	data_pth = '{}/data/{}/{}'.format(pth,state,action)

	return data_pth, model_pth, plot_pth