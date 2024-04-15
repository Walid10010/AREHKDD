

import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(6, 6))
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 200
from sklearn.metrics import roc_auc_score
from pyod.models.iforest import IForest


import matplotlib.pyplot as plt


X, Y = np.loadtxt('rebuttaldata.csv'), np.loadtxt('rebuttallabel.csv')

# np.savetxt('rebuttaldata.csv', X), np.savetxt('rebuttallabel.csv', Y)
a = plt.scatter(X[Y==0,0], X[Y==0,1])
b = plt.scatter(X[Y==1,0], X[Y==1,1])

random_seeds = [0, 1, 2, 3, 4, 5, 10, 100, 1000, 10000]
plt.legend((a,b), ('Normal', 'Outlier'))
plt.savefig('RebuttalKdd2024_remote_cluster.jpg')
# plt.axis('off')
plt.show()

iForest_roc = 0
for seed in random_seeds:
	algo = IForest(random_state=seed)
	algo.fit(X)
	y_score = algo.decision_function(X)
	iForest_roc += roc_auc_score(Y.reshape(-1), y_score) /len(random_seeds)

from ARDEH import ADERH


print('Iforest:ROC:', iForest_roc)

aderh_roc = 0

for seed in random_seeds:
	aderh = ADERH(random_state=seed)
	aderh.fit(X)
	outlier_scoer, label = aderh.decision_function(X), aderh.labels_
	aderh_roc += roc_auc_score(Y.reshape(-1), outlier_scoer) / len(random_seeds)



print('ADERH_ROC:', aderh_roc)
