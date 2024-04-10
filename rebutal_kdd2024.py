
# Gesamter Python-Code, um die angepasste Exponentialfunktion und lineare Funktion zu zeichnen

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(6, 6))
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
from sklearn.metrics import roc_auc_score
from pyod.models.iforest import IForest


import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, Y = make_blobs( n_samples=1000,centers=1, center_box=(-0.5, -0.5), random_state=0, cluster_std=0.1, )
X2, Y2= make_blobs(n_samples=1000, centers=1, center_box=(0, 0), random_state=0, cluster_std=0.1, )
X = np.append(X, X2, axis=0)
Y = np.array([0]*len(X))
X2 = np.array([[-0.85,-0.6], [-.12,-0.6],[-.6,-.15], [-.6,-.9], [-.4, -.95], [-0.85,-0.5], [0.3, 0.2], [0.2, 0.2], [-0.4, 0.15], [0.3, -.2] ])
X = np.append(X, X2, axis=0)
Y = np.append(Y, np.array([1]*len(X2)))
plt.scatter(X[:,0], X[:,1], edgecolors = 'k', c= Y.tolist())


algo = IForest(random_state=0)
algo.fit(X)
y_score = algo.decision_function(X)
print(roc_auc_score(Y.reshape(-1), y_score))
from ARDEH import ADERH

aderh = ADERH(random_state=0)
aderh.fit(X)
outlier_scoer, label = aderh.decision_function(X), aderh.labels_
print(roc_auc_score(Y.reshape(-1), outlier_scoer))

plt.show()