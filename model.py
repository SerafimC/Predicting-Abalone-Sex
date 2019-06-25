import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

#Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


from data_prep import X_train, X_test, y_train, y_test, target_names

clf = SVC(decision_function_shape= 'ovr',
			gamma='auto', 
			# tol=1e-20,
			# class_weight='balanced', 
			kernel='rbf', 
			# random_state=0,
			# verbose=False,
			# probability=False,
			C=1)

clf.fit(X_train, y_train)

y_hat = clf.predict(X_test)

print(metrics.classification_report(y_test, y_hat, target_names=target_names))
print('Acerto: ' + str(np.mean(y_hat == y_test)*100) + '%') 

#=========================================================
# KMEANS and DBSCAN
#=========================================================
# clusterer = KMeans(n_clusters=3, random_state=10, n_init=100, tol=1e-6)
# clusterer.fit(X_train)

# y_hat = clusterer.predict(X_test)

# print(metrics.classification_report(y_test, y_hat, target_names=target_names))
# print('Acerto KMeans: ' + str(np.mean(y_hat == y_test)*100) + '%') 

# clusterer = DBSCAN(eps=5, min_samples=30)
# clusterer.fit(X_train)

# y_hat = clusterer.fit_predict(X_test)

# # print(metrics.classification_report(y_test, y_hat, target_names=target_names))
# print('Acerto DBSCAN: ' + str(np.mean(y_hat == y_test)*100) + '%') 

#=========================================================
# Decision tree to get the most significant features
#=========================================================
# clf = DecisionTreeClassifier(random_state=0)
# clf.fit(X_train, y_train)
# print(cross_val_score(clf, X_test, y_test, cv=10))