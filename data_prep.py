import statistics
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import preprocessing

abalone = np.genfromtxt('abalone.data', delimiter=',', dtype='str')

#labels
targets = abalone[:, 0]

le = preprocessing.LabelEncoder()
le.fit(targets)

targets = le.transform(targets)

# Remove labels
abalone = np.delete(abalone, 0, axis=1)

# convert to float all data
abalone = list(map(lambda x: [float(s) for s in x], abalone))
abalone = np.asarray(abalone)

# Features importance
# [0.09266842, 0.07043336, 0.08531505, 0.13925405, 0.13836344, 0.27438111, 0.1042474 , 0.09533717]

# Features
features = abalone

for i in range(features.shape[0]):
    # if targets[i] == 2:
        for j in range(features.shape[1]):
            mean = np.mean(features[:, j])
            features[i, j] = ((features[i, j] / mean) * features[i, j]) + round(features[i, j] - mean,2)*10
            # if j == 0:
            #     features[i, j] = (features[i, j]/mean)
            # else:
            #     features[i, j] = (features[i, j]/mean) * features[i, j-1] 
            
            #print("Desvio Padrao: " + str(statistics.pstdev(features[:, j])))
            # features[i, j] = ((features[i, j] / mean) * features[i, j])**2


# pca = PCA(n_components = 4)
# features = pca.fit_transform(abalone)

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.3, random_state=0)

target_names = np.array(['F', 'I', 'M'])
feature_names = np.array(['len', 'diameter', 'height', 'Whole weight',
                'Shucked weight', 'Viscera weight',
                'Shell weight', 'Rings'])