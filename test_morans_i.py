# %%
import numpy as np
import pandas as pd

def morans_i(distance_matrix, values):

    # Number of observations
    N = len(values)
    
    # Mean of the values
    mean_value = np.mean(values)
    
    # Differences from the mean
    diff = values - mean_value
    
    # Weights matrix
    W = np.sum(distance_matrix)
    
    # Calculate the numerator
    numerator = 0
    for i in range(N):
        for j in range(N):
            numerator += distance_matrix[i, j] * diff[i] * diff[j]
    
    # Calculate the denominator
    denominator = np.sum(diff ** 2)
    
    # Moran's I
    I = (N / W) * (numerator / denominator)
    
    return I


dist_mat = np.load('data/template_parc-Schaefer400_desc-distance.npy')
weight = 1 / dist_mat

