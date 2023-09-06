import numpy as np
from collections import defaultdict
from carton_env import CartonEnv
from math import comb
from itertools import combinations


class EggAgent():

    '''
    The Q-learning agent that solves the egg-carton environment

    Args:
        rows (int): number of rows in carton
        cols (int): number of columns in carton
        num_actions (int): number of eggs to take out at each time step
        (rows * cols must be divisible by num_actions)
    '''

    def __init__(self, rows=2, cols=6, num_actions=3):
        # Initialize params
        self.num_actions = num_actions
        self.alpha = 0.34
        self.gamma = 0.9
        self.epsilon = 0.18
        self.n_episodes = 300000

        # Initialize environment and rand seed
        self.env = CartonEnv(rows=rows, cols=cols, num_actions=num_actions)

        # Initialize Q(s,a)
        # Q[state][action(s)] = value
        # Each action_id is a tuple of egg_ids
        # Therefore there are num_eggs choose num_actions possible actions
        self.actions = combinations(self.env.action_space, num_actions)
        self.total_actions = len(self.actions)
        self.Q = defaultdict(lambda: np.zeros(self.total_actions))

        np.random.seed(123)

    def solve(self):
        '''
        Create the Q table
        '''
        
        # Implement Q-learning
        for _ in range(self.n_episodes):
            state = self.env.reset()
            while done is False:
                # Choose action using policy derived from Q (epsilon-greedy)
                rand = np.random.random_sample()
                if rand < self.epsilon:
                    action_id = np.random.randint(self.total_actions)
                else:
                    # Choose greediest action
                    action_id = np.argmax(self.Q[state])
                    
                # Take action and observe R, S'
                action = self.actions[action_id]
                state_next, reward, done = self.env.step(action)
                
                # Update Q
                max_Q = np.amax(self.Q[state_next])
                td_error = reward + (self.gamma * max_Q) - self.Q[state][action_id]
                self.Q[state][action_id] += self.alpha * td_error
                state = state_next
