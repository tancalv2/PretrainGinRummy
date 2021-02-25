import os
import csv

class Logger(object):
    ''' Logger saves the running results and helps make plots from the results
    '''

    def __init__(self, log_dir, env=''):
        ''' Initialize the labels, legend and paths of the plot and log file.

        Args:
            log_path (str): The path the log files
        '''
        self.log_dir = log_dir
        self.txt_path = os.path.join(log_dir, env + 'log.txt')
        self.csv_path = os.path.join(log_dir, env + 'performance.csv')
        self.fig_path = os.path.join(log_dir, env + 'fig.png')
        # plot avg turns vs. timestep
        self.fig_path2 = os.path.join(log_dir, env + 'fig2.png')
        # plot cond(knock) vs. timestep
        self.fig_path3 = os.path.join(log_dir, env + 'fig3.png')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.txt_file = open(self.txt_path, 'w')
        self.csv_file = open(self.csv_path, 'w')
        fieldnames = ['timestep', 'reward']
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        # self.writer.writeheader()

        fieldnames2 = ['timestep', 'reward', 'Avg Turns', 'Cond_Knock']
        self.writer2 = csv.DictWriter(self.csv_file, fieldnames=fieldnames2)
        self.writer2.writeheader()

    def log(self, text):
        ''' Write the text to log file then print it.
        Args:
            text(string): text to log
        '''
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        print(text)

    def log_performance(self, timestep, reward):
        ''' Log a point in the curve
        Args:
            timestep (int): the timestep of the current point
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'timestep': timestep, 'reward': reward})
        print('')
        self.log('----------------------------------------')
        self.log('  timestep     |  ' + str(timestep))
        self.log('  reward       |  ' + str(reward))
        self.log('----------------------------------------')
    
    # 2021-02-22: Add actions into logs
    def log_performance2(self, timestep, reward, actions):
        ''' Log a point in the curve
        Args:
            timestep (int): the timestep of the current point
            reward (float): the reward of the current point
            actions (dict): the actions at the current point
        '''
        Cond_Knock = actions['Knock'] / actions['Knock_Possible'] if actions['Knock_Possible'] > 0  else 0
        self.writer2.writerow({'timestep': timestep, 'reward': reward,
                               'Avg Turns': actions['Avg Turns'], 'Cond_Knock': Cond_Knock})
        print('')
        self.log('----------------------------------------')
        self.log('  timestep     |  ' + str(timestep))
        self.log('  reward       |  ' + str(reward))
        self.log('  Avg Turns    |  ' + str(actions['Avg Turns']))
        self.log('  Cond_Knock   |  ' + str(Cond_Knock))
        self.log('  Gin: ' + str(actions['Gin']) + ' | Knock: ' + str(actions['Knock']) + ' | Other: ' + str(actions['Other']))
        self.log('----------------------------------------')

    def plot(self, algorithm):
        plot(self.csv_path, self.fig_path, algorithm)

    def plot2(self, algorithm):
        plot(self.csv_path, self.fig_path, algorithm, ylabel='reward')
        plot(self.csv_path, self.fig_path2, algorithm, ylabel='Avg Turns')
        plot(self.csv_path, self.fig_path3, algorithm, ylabel='Cond_Knock')

    def close_files(self):
        ''' Close the created file objects
        '''
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()

def plot(csv_path, save_path, algorithm, ylabel='reward'):
    ''' Read data from csv file and plot the results
    '''
    import matplotlib.pyplot as plt
    with open(csv_path) as csvfile:
        print(csv_path)
        reader = csv.DictReader(csvfile)
        xs = []
        ys = []
        for row in reader:
            xs.append(int(row['timestep']))
            ys.append(float(row[ylabel]))
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=algorithm)
        ax.set(xlabel='timestep', ylabel=ylabel)
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)
