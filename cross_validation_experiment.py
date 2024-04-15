from pyod.models.inne import INNE
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.lof import LOF
from pyod.models.loda import LODA
#from pyod.models.deep_svdd import DeepSVDD
#from pyod.models.so_gaal import SO_GAAL
from pyod.models.ecod import ECOD
from pyod.models.ocsvm import OCSVM
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.ocsvm import OCSVM

from sklearn.cluster import DBSCAN
from deepod.models.tabular import RDP, RCA
from ARDEH import ADERH
from sklearn.metrics import roc_auc_score
import glob
from sklearn.model_selection import StratifiedShuffleSplit


algo_dic ={'ADERH':ADERH, 'INNE':INNE, 'IForest':IForest, 'LOF':LOF, 'LODA': LODA, 'COPOD':COPOD, 'ECOD':ECOD}

import glob
import numpy as np

sss = StratifiedShuffleSplit(n_splits=3, test_size=0.3, random_state=0)

random_seeds = [0, 1, 2, 1000, 10000]

for data_name in glob.glob('Dataset/*'):
    if 'donor'   in data_name: continue
    print(data_name)
    if '9_c' in data_name:continue
    tt = False
    for algo_name in algo_dic:

        if tt: continue
        data = np.load('{}'.format(data_name),
                   allow_pickle=True)
        X, Y = data['X'], data['y']
        roc_score_value = 0

        if algo_name in (['ADERH', 'INNE', 'IForest', 'DIF']):
            for seed in random_seeds:
             for i, (train_index, test_index) in enumerate(sss.split(X, Y)):
                 algo = algo_dic[algo_name](random_state=seed)
                 algo.fit(X[train_index])
                 outlier_score, label = algo.decision_function(X[test_index]), algo.labels_
                 roc_score_value += roc_auc_score(Y[test_index], outlier_score) / (len(random_seeds) * 3)


        elif  algo_name in  ['RCA', 'RDP'] :
            algo = algo_dic[algo_name](device='cpu')
            algo.fit(X)
            outlier_score, label = algo.decision_function(X), algo.predict(X)
            roc_score_value = roc_auc_score(Y, outlier_score)

        elif  algo_name in ['DBSCAN']:
            algo = DBSCAN()
            y_pred = y_prob = algo.fit_predict(X)
            y_pred[y_pred >= 0] = -2
            y_pred[y_pred == -1] = 1
            y_pred[y_pred == -2] = 0
            roc_score_value = roc_auc_score(Y, y_pred)

        else:
            for i, (train_index, test_index) in enumerate(sss.split(X, Y)):
             algo = algo_dic[algo_name]()
             algo.fit(X[train_index])
             outlier_score, label = algo.decision_function(X[test_index]), algo.labels_
             roc_score_value += roc_auc_score(Y[test_index], outlier_score)/3

        print('dataset:{}, algo: {}, roc-score: {} '.format(data_name.split('/')[-1].split('.')[0], algo_name,   roc_score_value))




