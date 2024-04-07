import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import time



# class VariableNeighborhoodSearch:
#     def __init__(self, classifier, init_solution, current_solution):
#         self.classifier = classifier
#         self.init_solution = init_solution
#         self.current_solution = current_solution


#     def generate_N1(self):
#         neighbor_solutions = []
#         for i in range(len(self.current_solution)):
#             neighbor = list(self.current_solution)
#             neighbor[i] = 1 - neighbor[i]
#             neighbor_solutions.append(neighbor)
#         return neighbor_solutions
    
#     def generate_N2(self):
#         neighbor_solutions = []
#         for i in range(len(self.current_solution) - 1):
#             for j in range(i+1, len(self.current_solution)):
#                 neighbor = list(self.current_solution)
#                 neighbor[i] = 1- neighbor[i]
#                 neighbor[j] = 1 - neighbor[j]
#                 neighbor_solutions.append(neighbor_structures)
#         return neighbor_solutions
    
#     def g
            
def generate_N1(current_solution):
    neighbor_solutions = []
    for i in range(len(current_solution)):
        neighbor = list(current_solution)
        neighbor[i] = 1 - neighbor[i]
        neighbor_solutions.append(neighbor)
    return neighbor_solutions

def generate_neighbor_N2(current_solution):
    neighbor_solutions = []
    
    # Iterate through each pair of positions in the current solution
    for i in range(len(current_solution) - 1):
        for j in range(i + 1, len(current_solution)):
            # Create a copy of the current solution to modify
            neighbor = list(current_solution)
            
            # Flip the bits at positions i and j (0 to 1 or 1 to 0)
            neighbor[i] = 1 - neighbor[i]
            neighbor[j] = 1 - neighbor[j]
            
            # Add the modified neighbor solution to the list
            neighbor_solutions.append(neighbor)
    
    return neighbor_solutions

def generate_neighbor_N3(current_solution):
    neighbor_solutions = []
    
    # Iterate through each triplet of positions in the current solution
    for i in range(len(current_solution) - 2):
        for j in range(i + 1, len(current_solution) - 1):
            for k in range(j + 1, len(current_solution)):
                # Create a copy of the current solution to modify
                neighbor = list(current_solution)
                
                # Flip the bits at positions i, j, and k (0 to 1 or 1 to 0)
                neighbor[i] = 1 - neighbor[i]
                neighbor[j] = 1 - neighbor[j]
                neighbor[k] = 1 - neighbor[k]
                
                # Add the modified neighbor solution to the list
                neighbor_solutions.append(neighbor)
    return neighbor_solutions
import random

def generate_neighbor_N4(current_solution):
    neighbor = list(current_solution)
    index_to_change = random.randint(0, len(neighbor)-1)
    neighbor[index_to_change] = 1 - neighbor[index_to_change]
    return neighbor

def generate_neighbor_N5(current_solution):
    neighbor = list(current_solution)
    indices_to_change = random.sample(range(0, len(neighbor)), 2)  # Change 2 random indices
    for index in indices_to_change:
        neighbor[index] = 1 - neighbor[index]
    return neighbor

def generate_neighbor_N6(current_solution):
    neighbor = list(current_solution)
    indices_to_change = random.sample(range(0, len(neighbor)), 3)  # Change 3 random indices
    for index in indices_to_change:
        neighbor[index] = 1 - neighbor[index]
    return neighbor

def generate_neighbor_N7(current_solution):
    neighbor = list(current_solution)
    indices_to_change = random.sample(range(0, len(neighbor)), 4)  # Change 4 random indices
    for index in indices_to_change:
        neighbor[index] = 1 - neighbor[index]
    return neighbor

def generate_neighbor_N8(current_solution):
    neighbor = list(current_solution)
    indices_to_change = random.sample(range(0, len(neighbor)), 5)  # Change 5 random indices
    for index in indices_to_change:
        neighbor[index] = 1 - neighbor[index]
    return neighbor


accuracy_cache = {} # Hash table to store computed accuracy values
def tune_parameters(classifier, param_grids):
    grid_search = GridSearchCV(classifier, param_grids, cv= 5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    return best_params

"""Parameters tuning (Grid Search) => VNS - feature seletion
=> Train => Test => Comparison against imcumbent solution
=> Selection of the best => Repeat VNS until the num of iterations
=> VNS result """
"""


VNS - Variable Neighborhood Search
Input: a set of neighborhood strutures N_k for 
k = 1, 2, ... k_max and a initial solution x

    Repeat
        k = 1
            Repeat  
                Shaking: choose a random solution x_r from N_k(x)
                x_i = localSearch(x_r)
                If f(x_i) > f(x) Then
                    x = x_i
                    k = 1
                Else k = k+1
            Until k = k_max
    Until Stopping criteria (iterations = 100)
Output: Best found solution
"""

def calculate_acc(x):
    selected_features = define_selected_features(x)
    print("select features: ", selected_features)
    # print(X_train.iloc[:,selected_features].shape)
    best_model.fit(X_train.iloc[:,selected_features], y_train)
    y_preds = best_model.predict(X_test.iloc[:, selected_features])
    # print("y_test: ", y_test.values)
    # print("y preds:", y_preds)
    return accuracy_score(y_test.values, y_preds.reshape(-1, 1))

def define_selected_features(x):
    x = [index for index, values in enumerate(x) if values == 1]
    return x

def variable_neighborhood_search(neighborhood_structures, init_solution, iterations = 100):
    x = init_solution
    best_solution = x
    iter = 0
    while iter < iterations:
        k = 1
        print("iter: ", iter)
        while k < len(neighborhood_structures):
            x_r = random.choice(neighborhood_structures[k](x))
            if x_r == 0*len(X.columns):
                x_r = x
            else:
                if len(neighborhood_structures[k](x)) == len(X.columns):
                    x_r = neighborhood_structures[k](x)
                    x_i = localSearch(x_r, neighborhood_structures[k])

                x_i = localSearch(x_r, neighborhood_structures[k])
                print("X_i", x_i)
                if calculate_acc(x_i)> calculate_acc(x):
                    x = x_i
                    print("x: ", x)

                    best_solution = x
                    k = 1
                else: k +=1
        iter +=1
    return best_solution

    
def update_acc(x):
    key = tuple(x)
    if key in accuracy_cache:
         return accuracy_cache[key]
    acc = calculate_acc(x)
    accuracy_cache[key] = acc
    return acc
    
def localSearch(x_r, neighborhood_structure):
    best_neighbor = x_r
    print("best neighbor:", x_r)
    best_acc = calculate_acc(x_r)
    print("Acc: ", best_acc)
    
    if len(neighborhood_structure(x_r)) == len(X.columns):
        neighbor = neighborhood_structure(x_r)
        acc = calculate_acc(neighbor)
        if acc > best_acc:
            best_neighbor = neighbor
            best_acc = acc
    else:
        for neighbor in neighborhood_structure(x_r):
            if neighbor == 0*len(X.columns):
                continue
            print("neighbor:", neighbor)
            acc = calculate_acc(neighbor)
            if acc > best_acc:
                best_neighbor = neighbor
                best_acc = acc
    return best_neighbor

def generate_neighbor_N1(current_solution):
    neighbor_solutions = []
    # Iterate through each position in the current solution
    for i in range(len(current_solution)):
        # Create a copy of the current solution to modify
        neighbor = list(current_solution)
        
        # Flip the bit at position i (0 to 1 or 1 to 0)
        neighbor[i] = 1 - neighbor[i]
        
        # Add the modified neighbor solution to the list
        neighbor_solutions.append(neighbor)
    
    return neighbor_solutions

def generate_neighbor_N2(current_solution):
    neighbor_solutions = []
    
    # Iterate through each pair of positions in the current solution
    for i in range(len(current_solution) - 1):
        for j in range(i + 1, len(current_solution)):
            # Create a copy of the current solution to modify
            neighbor = list(current_solution)
            
            # Flip the bits at positions i and j (0 to 1 or 1 to 0)
            neighbor[i] = 1 - neighbor[i]
            neighbor[j] = 1 - neighbor[j]
            
            # Add the modified neighbor solution to the list
            neighbor_solutions.append(neighbor)
    
    return neighbor_solutions

def generate_neighbor_N3(current_solution):
    neighbor_solutions = []
    
    # Iterate through each triplet of positions in the current solution
    for i in range(len(current_solution) - 2):
        for j in range(i + 1, len(current_solution) - 1):
            for k in range(j + 1, len(current_solution)):
                # Create a copy of the current solution to modify
                neighbor = list(current_solution)
                
                # Flip the bits at positions i, j, and k (0 to 1 or 1 to 0)
                neighbor[i] = 1 - neighbor[i]
                neighbor[j] = 1 - neighbor[j]
                neighbor[k] = 1 - neighbor[k]
                
                # Add the modified neighbor solution to the list
                neighbor_solutions.append(neighbor)
    return neighbor_solutions
import random

def generate_neighbor_N4(current_solution):
    neighbor = list(current_solution)
    index_to_change = random.randint(0, len(neighbor)-1)
    neighbor[index_to_change] = 1 - neighbor[index_to_change]
    return neighbor

def generate_neighbor_N5(current_solution):
    neighbor = list(current_solution)
    indices_to_change = random.sample(range(0, len(neighbor)), 2)  # Change 2 random indices
    for index in indices_to_change:
        neighbor[index] = 1 - neighbor[index]
    return neighbor

def generate_neighbor_N6(current_solution):
    neighbor = list(current_solution)
    indices_to_change = random.sample(range(0, len(neighbor)), 3)  # Change 3 random indices
    for index in indices_to_change:
        neighbor[index] = 1 - neighbor[index]
    return neighbor

def generate_neighbor_N7(current_solution):
    neighbor = list(current_solution)
    indices_to_change = random.sample(range(0, len(neighbor)), 4)  # Change 4 random indices
    for index in indices_to_change:
        neighbor[index] = 1 - neighbor[index]
    return neighbor

def generate_neighbor_N8(current_solution):
    neighbor = list(current_solution)
    indices_to_change = random.sample(range(0, len(neighbor)), 5)  # Change 5 random indices
    for index in indices_to_change:
        neighbor[index] = 1 - neighbor[index]
    return neighbor


neighbor_structures = [generate_neighbor_N1, generate_neighbor_N2, 
                       generate_neighbor_N3, generate_neighbor_N4, 
                       generate_neighbor_N5, generate_neighbor_N6,
                       generate_neighbor_N7, generate_neighbor_N8 ]


if __name__ == "__main__":
    start_time = time.time()
    dataset_path = "/home/cuctt/credit/data/australian/australian.dat"
    data = pd.read_csv(dataset_path, delimiter=" ", header= None)
    X = data.iloc[:, :-1]  # Features (all columns except the last one)
    y = data.iloc[:, -1] # Target (last column)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state=  42)
    print(y_test.values)
    classifier = LogisticRegression(max_iter= 10000)
    param_grids = {
        "C": [0.1, 1.0, 10.0],
        "penalty": ["l1", "l2"]}
    best_params = tune_parameters(classifier, param_grids)
    best_model = LogisticRegression(**best_params)
    incumbent_solution = [1] * len(X.columns)
    best_selected_features = variable_neighborhood_search(neighbor_structures, incumbent_solution)
    print(best_selected_features)
    selected_features = define_selected_features(best_selected_features)
    end_time = time.time()
    print("Optimize time:", end_time - start_time)
    best_model.fit(X_train.iloc[:,selected_features], y_train)
    y_pred = best_model.predict(X_test.iloc[:, selected_features])
    accuracy = accuracy_score(y_test.values, y_pred.reshape(-1,1))
    print(accuracy)



    