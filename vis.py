import numpy as np
import matplotlib.pyplot as plt
from data_prep import features, targets, target_names, feature_names

data = features

estilo = ['b', 'r', 'g', 'y']

for i in range(feature_names.shape[0]):
    for j in range(feature_names.shape[0]):
        if i == j or j<i:
            continue
        plt.scatter(data[targets == 0, i], data[targets == 0, j], 1, color=estilo[0], label=target_names[0])
        plt.scatter(data[targets == 1, i], data[targets == 1, j], 1, color=estilo[1],label=target_names[1])
        plt.scatter(data[targets == 2, i], data[targets == 2, j], 1, color=estilo[2],label=target_names[2])
        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])
        plt.legend()
        plt.show()