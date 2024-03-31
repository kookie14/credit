import numpy as np
import pandas as pd
import random
import time
import warnings 
warnings.filterwarnings("ignore")

class Neighborhood():
    def __init__(self, init_solution, current_solution):
        self.init_solution = init_solution
        self.current_solution = current_solution
        self.num_feature = len(self.current_solution)
        self.neighborhood_solution = list()

    def neighbor_1(self):
        neigbors = list(self.current_solution)
        for x in range(self.num_feature):
            neigbors[x] = 1 - neigbors[x]
            self.neighborhood_solution.append(neigbors[x])
        
    

        