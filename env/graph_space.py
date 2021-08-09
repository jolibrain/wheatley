# This was a test to see if we can make a custom gym.spaces.Space, in order to be able
# to use graph. Seems like we can't use stable_baselines3 if we do that, since they
# check the type of the Space, and if not in allowed types, they throw an error.

import gym
from gym.spaces.space import Space
import numpy as np
import torch
from torch_geometric.data import Data

class GraphSpace(Space):

    def __init__(self, n_features):
        super(GraphSpace, self).__init__((), np.float32)
        self.n_features = n_features

    def sample(self):
        """ 
        Create a random graph, with a number of nodes between 1 and 10 and no edges
        """
        n_nodes = np.randint(10) + 1
        features = torch.tensor(n_nodes, self.n_features)
        return Data(x=features, edge_index=None)

    def contains(self, x):
        if isinstance(x, Data):
            return True
        return False
