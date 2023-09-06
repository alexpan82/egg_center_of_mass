import numpy as np
import itertools
from collections import defaultdict


class CartonEnv():

    '''
    Sets up egg carton env

    Args:
        rows (int): number of rows in carton
        cols (int): number of columns in carton
        num_actions (int): number of eggs to take out at each time step
        (rows * cols must be divisible by num_actions)

        Each egg is associated with an egg_id ie (x, y) position
    '''

    def __init__(self, rows=2, cols=6, num_actions=3):
        self.rows = rows
        self.cols = cols
        self.num_actions = num_actions

        # Represent eggs as 1's and empty slots in carton as 0's
        self.current_carton = np.ones((self.rows, self.cols))

        # Let position (x=0, y=0) be at the bottom left corner of the rectangle with size rows x cols
        # Let an egg lie at (x=0, y=0). All other eggs lie at the centers a square grid with edge length = 1
        # Find the COM with respect to (x=0, y=0)
        self.x_position = np.arange(self.cols)
        self.y_position = np.arange(self.rows)

        # Find center of mass (COM)
        self.current_com = self.calc_com()

        # To bound the reward, consider the maximum possible change in COM
        self.max_com_diff = np.linalg.norm(np.array([self.cols - 1, self.rows - 1]))

        # Each action is to remove a egg(s)
        self.action_space = map(tuple, list(itertools.product(self.x_position, self.y_position)))
        self.action_space = list(self.action_space)

        # Keep track of which eggs were removed
        # removed_eggs[(x, y)] = 0 or 1
        self.removed_eggs = defaultdict(lambda: 0)


    def calc_com(self):
        # Find the center of mass (COM) of current_carton
        # Let eggs have mass = 1
        total_mass = np.sum(self.current_carton)

        com_x = np.sum(self.current_carton * self.x_position) / total_mass
        com_y = np.sum(self.current_carton.T * self.y_position) / total_mass

        return np.array([com_x, com_y])
    

    def reset(self):
        self.current_carton = np.ones((self.rows, self.cols))
        self.current_com = self.calc_com()
        self.removed_eggs = defaultdict(lambda: 0)
        return self.current_carton


    def step(self, action):
        # action is a list with size num_actions corresponding to which egg_ids are to be removed
        old_com = self.current_com

        if self._is_valid_action(action):
            print('hi', action)
            for egg_id in action:
                # x, y = self.action_space[egg_id]
                x, y = egg_id
                self.current_carton[x, y] = 0
            self.current_com = self.calc_com()

        # Done if all eggs are removed
        if np.sum(self.current_carton) == 0:
            done = True
        else:
            done = False
        
        # Reward
        # Penalize if the carton remains the same since this means an invalid action was taken
        # Should scale with size of carton since we want the penalization to be fair
        if not self._is_valid_action(action):
            reward = -1 * self.rows * self.cols

        # Reward proportional to how much the COM has moved
        else:
            # Find euclidean dist
            com_diff_dist = np.linalg.norm(self.current_com - old_com)
            reward = self.max_com_diff - com_diff_dist

        return self.current_carton, reward, done


    def _is_valid_action(self, action):
        # action is a list with size num_actions
        # Cannot remove an egg that has already been removed
        # or is currently being removed
        for a in action:
            if self.removed_eggs[a] != 0:
                return False
        
        if len(set(action)) != self.num_actions:
            return False
        
        return True