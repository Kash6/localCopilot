decision_tree_functions.py:
# coding: utf-8

import numpy as np
import pandas as pd
import random

from helper_functions import determine_type_of_feature


# 1. Decision Tree helper functions 
# (see "decision tree algorithm flow chart.png")

# 1.1 Data pure?
def check_purity(data):
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False

    
# 1.2 Classify
def classify_data(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification


# 1.3 Potential splits?
def get_potential_splits(data, random_subspace):
    
    potential_splits = {}
    _, n_columns = data.shape
    column_indices = list(range(n_columns - 1))    # excluding the last column which is the label
    
    if random_subspace and random_subspace <= len(column_indices):
        column_indices = random.sample(population=column_indices, k=random_subspace)
    
    for column_index in column_indices:          
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
    
    return potential_splits


# 1.4 Lowest Overall Entropy?
def calculate_entropy(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
    
    entropy = 0.0
    for count in counts_unique_classes:
        prob_class = count / len(label_column)
        entropy += - prob_class * np.log2(prob_class)
    
    return entropy


# 1.5 Decision Tree
def decision_tree(data, random_subspace=None):
    
    if check_purity(data):
        return classify_data(data)
    
    potential_splits = get_potential_splits(data, random_subspace)
    best_gain = -1
    best_column = None
    
    for column_index, values in potential_splits.items():
        entropy_splits = []
        for value in values:
            data_left = data[data[:, column_index] < value]
            data_right = data[data[:, column_index] >= value]
            
            entropy_left = calculate_entropy(data_left)
            entropy_right = calculate_entropy(data_right)
            
            entropy_split = (len(data_left) / len(data)) * entropy_left + (len(data_right) / len(data)) * entropy_right
            entropy_splits.append(entropy_split)
        
        gain = calculate_entropy(data) - np.mean(entropy_splits)
        
        if gain > best_gain:
            best_gain = gain
            best_column = column_index
    
    best_column = potential_splits.keys()[best_column]
    
    tree = {best_column: {}}
    
    for value in potential_splits[best_column]:
        sub_data = data[data[:, best_column] < value]
        sub_tree = decision_tree(sub_data