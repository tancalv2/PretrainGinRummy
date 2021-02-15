import subprocess
import sys
from distutils.version import LooseVersion

reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]

if 'tensorflow' in installed_packages:
    import tensorflow as tf
    if LooseVersion(tf.__version__) < LooseVersion('1.14.0') \
            or LooseVersion(tf.__version__) >= LooseVersion('2.0.0'):
        print('WAINING - RLCard supports Tensorflow >=1.14 and <2.0\nThe detected version is {} \nIf the models can not be loaded, please install Tensorflow via\n$ pip install rlcard[tensorflow]\n'.format(tf.__version__))
    from rlcard.agents.dqn_agent import DQNAgent
if 'torch' in installed_packages:
    from rlcard.agents.dqn_agent_pytorch import DQNAgent as DQNAgentPytorch

from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.simpleGinRummy_agent import SimpleGinRummyAgent