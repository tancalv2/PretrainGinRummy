import os
from shutil import copyfile

# Available states-action pairs:
state_action_pair = {'all': 'all', # all actions
                    'bpbd': 'draw', # actions 2/3 
                    'apbd': ['discard', 'knock'], # actions 6-57, 58-109
                    'apad': 'knock_bin'} # binary action

# All Possible Classes
all_classes = ['SP0','SP1','Draw','Pickup','DH','GIN',
               'AS', '2S', '3S', '4S', '5S', '6S', '7S', '8S', '9S', 'TS', 'JS', 'QS', 'KS',
               'AH', '2H', '3H', '4H', '5H', '6H', '7H', '8H', '9H', 'TH', 'JH', 'QH', 'KH',
               'AD', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', 'TD', 'JD', 'QD', 'KD',
               'AC', '2C', '3C', '4C', '5C', '6C', '7C', '8C', '9C', 'TC', 'JC', 'QC', 'KC',
               'AS', '2S', '3S', '4S', '5S', '6S', '7S', '8S', '9S', 'TS', 'JS', 'QS', 'KS',
               'AH', '2H', '3H', '4H', '5H', '6H', '7H', '8H', '9H', 'TH', 'JH', 'QH', 'KH',
               'AD', '2D', '3D', '4D', '5D', '6D', '7D', '8D', '9D', 'TD', 'JD', 'QD', 'KD',
               'AC', '2C', '3C', '4C', '5C', '6C', '7C', '8C', '9C', 'TC', 'JC', 'QC', 'KC']

# Class Groups and start/stop indices
class_groups = {'draw': {'ind': [2,4], 
                         'classes': ['Draw/Pickup', 'Other']},
                'discard': {'ind': [6,58],
                            'classes': ['Discard Action', 'Other']},
                'knock': {'ind': [58,110],
                          'classes': ['Knock Action', 'Other']}}


################################################# Create Directories #################################################
def create_dir(pth, state, action, model_name):
	'''
	create model and plot directories if do not exist
    '''
	if state in ['all','bpbd','apbd','apad']:
		if action in state_action_pair[state]:
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
			print('Directories created.')

			return data_pth, model_pth, plot_pth
		else:	
			print('illegal state-action pair (action)')
			return None, None, None
	else:
		print('illegal state-action pair (state)')
		return None, None, None


################################################# Write Parameters #################################################
def write_params(pth, state, action, model_name, numGames, pruneStatesList, actionChoice, balance, batch_size, lr, epochs, model_fnc, activation, loss, loss_weight, pre_train, model_PT, device):
	
	model_pth = '{}/models/{}'.format(pth,state)
	model_pth = '{}/{}'.format(model_pth,action)
	model_pth = '{}/{}.txt'.format(model_pth,model_name)

	plot_pth = '{}/plots/{}'.format(pth,state)
	plot_pth = '{}/{}'.format(plot_pth,action)
	plot_pth = '{}/{}.txt'.format(plot_pth,model_name)

	# Delete old version from model directory
	if os.path.exists(model_pth):
  		os.remove(model_pth)

  	# Write params into txt file (add onto list as parameters increase)
	with open(model_pth, 'w') as f:
		f.write('State: {}\n'.format(state))
		f.write('Action: {}\n\n'.format(action))
		f.write('model_name: {}\n\n'.format(model_name))
		f.write('numGames: {}\n\n'.format(numGames))
		f.write('pruneStatesList: {}\n'.format(pruneStatesList))
		f.write('actionChoice: {}\n'.format(actionChoice))
		f.write('balance: {}\n\n'.format(balance))
		f.write('batch_size: {}\n'.format(batch_size))
		f.write('lr: {}\n'.format(lr))
		f.write('epochs: {}\n\n'.format(epochs))
		f.write('model_fnc: {}\n'.format(model_fnc))
		f.write('activation: {}\n\n'.format(activation))
		f.write('loss: {}\n'.format(loss))
		f.write('loss_weight: {}\n\n'.format(loss_weight))
		f.write('pre_train: {}\n'.format(pre_train))
		f.write('model_PT: {}\n\n'.format(model_PT))
		f.write('device: {}\n'.format(device))
		f.close()

	# Delete old version from plots directory
	if os.path.exists(plot_pth):
  		os.remove(plot_pth)

	# Copy params txt file to plots directory
	copyfile(model_pth, plot_pth)
	print('Parameters written.')
