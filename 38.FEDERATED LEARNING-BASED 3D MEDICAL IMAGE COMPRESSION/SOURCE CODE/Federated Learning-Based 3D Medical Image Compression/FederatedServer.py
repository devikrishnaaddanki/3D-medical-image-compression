import copy

import numpy as np

import logging
logger = logging.getLogger(__name__)

class FederatedServer:

    def __init__(self):
        self.global_weight = None
        self.local_weights = []    
        print("Federated init")

    def update(self, local_weight):
        if len(self.local_weights) < 2:
            self.local_weights.append(local_weight)            

    def avg(self):
        temp_list = []
        for i in range(len(self.local_weights)):
            for j in range(len(self.local_weights[i])):
                temp = np.array(self.local_weights[i][j])
                temp_list.append(temp)
        temp_list = np.asarray(temp_list)
        if len(self.local_weights) == 1:
            self.global_weight = np.divide(temp_list, len(self.local_weights))
        else:
            self.global_weight = np.divide(temp_list, len(self.local_weights)-1)
        return self.global_weight

    
