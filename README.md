# PretrainGinRummy
EngSci 2020-2021 Undergraduate Thesis: Pre-Initializing Q-Learning for Card Games with Large Feature Spaces

---
## Abstract
Card games with large feature spaces are often unlearnable by a Reinforcement Learning agent through common techniques like Q-Learning, especially when reward signals are sparse. Gin Rummy is such a card game that does not converge when training a Reinforcement Learning agent. An approach to allow an agent to learn these card games is though Pre-Initialization. Deep Q-Networks (DQNs), a branch of Q-Learning, consists of a Neural Network which can be Pre-Initialized through Supervised Learning. First, a Deep Neural Network model was trained on a supervised task of predicting a baseline agent’s action provided a state. The goal of the model is to learn lower-level features of the card game environment based on the baseline agent policy. Once the model sufficiently learned this policy, the model’s parameters are used as the initialization for the Reinforcement Learning agent. This, in theory, will allow the agent to focus on improving the learned features and policy instead of trying to learn both simultaneously. Multiple experiments were performed to the Pre-Initialized DQN agent including but not limited to layer freezing, varying reward structures, additional top layers, and additional action biases. Finally, these trained agents were evaluated against the baseline agent, and itself, both prior and during the training process, to assess the validity of the approach of Pre-Initializing the Deep Q-Network. Through these experiments, a final agent successfully improved during the DQN training process, supporting this approach to address Q-Learning convergence.

---
## Environments:
Ran using Google Colab Environment

### Supervised Learning
- Environment adapted from [here](https://github.com/AnthonyHein/GinRummyEAAI-Python) based on original Gin Rummy Challenge [here](https://github.com/tneller/gin-rummy-eaai).
- [Adapted Environment Code](https://github.com/tancalv2/PretrainGinRummy/tree/main/GinRummy)
- [Data Generation](https://github.com/tancalv2/PretrainGinRummy/blob/main/GenerateData.ipynb)
- [Model Generation Notebooks](https://github.com/tancalv2/PretrainGinRummy/tree/main/train)
    - [Model Generation Instructions](https://github.com/tancalv2/PretrainGinRummy/blob/main/GenerateModel.ipynb)
    -[Training Plots](https://github.com/tancalv2/PretrainGinRummy/tree/main/plots)
    -[Models](https://github.com/tancalv2/PretrainGinRummy/tree/main/models)

### Reinforcement Learning (DQN) Environment
- Environment adapted from [here](https://github.com/datamllab/rlcard)
- [Adapted Environment Code](https://github.com/tancalv2/PretrainGinRummy/tree/main/dqn)
- [DQN Training Notebooks](https://github.com/tancalv2/PretrainGinRummy/tree/main/train_dqn)
    -[Training Plots](https://github.com/tancalv2/PretrainGinRummy/tree/main/plots/dqn)
    -[Models](https://github.com/tancalv2/PretrainGinRummy/tree/main/models/dqn)

### Testing Environment
- Same Enviornment as Supervised Learning
- See Notebooks: [testAgents](https://github.com/tancalv2/PretrainGinRummy/blob/main/testAgents.ipynb) or [testAgents_final](https://github.com/tancalv2/PretrainGinRummy/blob/main/testAgents_final.ipynb)
