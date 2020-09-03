import random
import numpy as np
from collections import deque
import xgboost as xgb

class RLAgent:

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500000)   # agent memory
        self.epsilon = 1.0                   # exploration rate, decays during training
        self.epsilon_min = 0.0               
        self.epsilon_decay = 0.5
        self.model = None

    def remember(self, sar):
        self.memory.append(sar)

    def getMaxQ(self, state):

        if self.model is None:
            return 0.

        maxQ = -9999999.9 # so that this is automatically updated on first iteration
        
        # possible actions are [long, flat, short]
        for a in range(self.action_size): 

            np_state = np.reshape(state, [1, self.state_size])
            np_action = np.reshape(a, [1, 1])

            input = np.concatenate((np_state, np_action), axis=1)

            input_xgb = xgb.DMatrix(input)
            
            action_Q = self.model.predict(input_xgb)

            if action_Q[0] > maxQ:
                maxQ = action_Q[0]
                
        # return Q value for best action
        return maxQ

    def act(self, state, use_explo=True):
        """
        Takes a state and uses internal model to infer reward in policy space.
        Returns argmax of policy space.
        """
        
        # evaluate whether to perform optimally according to Q (exploit) - or explore
        # decaying epsilon greedy strategy
        if use_explo: # allow for stochastic exploration parameter
            if np.random.rand() < self.epsilon or self.model is None:
                return random.randrange(self.action_size)

        act_values=[]
        # for every action in action space
        for a in range(self.action_size):

            # take current sate and selected action
            np_state = np.reshape(state, [1, self.state_size])
            np_action = np.reshape(a, [1, 1]) # a = [0,1,2]

            # construct input from state,action pair
            input = np.concatenate((np_state, np_action), axis=1)

            # predict expected reward of action
            input_xgb = xgb.DMatrix(input)
            action_Q = self.model.predict(input_xgb)
            
            # append to list of Q values for actions
            act_values.append(action_Q)

        return np.argmax(act_values)  # returns optimal action

    def replay(self):
        # specify parameters via map
        max_depth = 6; num_round = 3; gamma = 0.05; eta = 0.75
        param = {'max_depth': max_depth, 'eta': eta, 'silent': 1, 'gamma': gamma, 'objective': 'reg:linear'}

        mem = np.array(self.memory)

        states  = mem[:, :self.state_size]                       # state dimension can be > 1
        actions = np.reshape(mem[:, self.state_size], [-1, 1])   # one action per state
        rewards = mem[:, -1]                                     # expected reward for each action given state

        # train an xgb model on current memory
        input_training = np.concatenate((states, actions), axis=1)
        training_data  = xgb.DMatrix(input_training, label=rewards)
        self.model     = xgb.train(param, training_data, num_round)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
