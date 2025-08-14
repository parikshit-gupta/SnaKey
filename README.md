# SnaKey

<img width="1000" height="582" alt="image" src="https://github.com/user-attachments/assets/b3943b17-f67c-4acb-8746-9003539ba35d" />

A Deep Q-Network based RL agent. Inspired from Google DeepMind's 2015 paper on DQNs with experience replay. Engineered a custom training environment from scratch enabling the agent to learn complex behaviors over time through strategic exploration and reward optimization. DQNs are a demonstration of the amalgamation of DL and RL. 

RL, originally inspired by how humans fundamentally learn all complex tasks: The reward system. Our sensory inputs are used by us to determine the next plausible action to be taken. The result might or might not be favourable but every (input, action, result) triplet tweaks our inter-neural connections, which helps in desicion making the next time we are in a similar situation. 

Using deep neural networks (here CNNs) to model Q-values for state action pairs of an environment violates 2 fundamental assumptions of DL: 1) Independent data points for training, 2) Stationarity of targets. These problems were alluded using sampling datapoints from an actively maintained replay memory and using a set of mirror weights with delayed updates respectively.

This agent has'nt been trained for an adequate amount of time due to computational constraints atm.

The folder "agent" contains the architecture of the CNN and the inner training loop. "environment" contains the custom gymnasium RL classic snake game env. "results" contains the final test code and some training instances.



