from pyod.models.inne import INNE
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.lof import LOF
from pyod.models.loda import LODA
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.ecod import ECOD
from pyod.models.ocsvm import OCSVM
from sklearn.cluster import DBSCAN
from deepod.models.tabular import RDP, RCA
from ARDEH import ADERH
from sklearn.metrics import roc_auc_score
import glob

# EIF: please use this version for EIF https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/eif.html
# PIDForest: https://github.com/vatsalsharan/pidforest

algo_dic ={'ADERH':ADERH, 'INNE':INNE, 'IForest':IForest, 'DIF':DIF, 'LOF':LOF, 'LODA': LODA, 'SO_GAAL':SO_GAAL, 'RCA':RCA, 'RDP':RDP, 'DBSCAN':DBSCAN, 'OCSVM':OCSVM }

import glob
import numpy as np


random_seeds = [0, 1, 2, 3, 4, 5, 10, 100, 1000, 10000]

for data_name in glob.glob('Dataset/*'):

    for algo_name in algo_dic:

        data = np.load('{}'.format(data_name),
                   allow_pickle=True)
        X, Y = data['X'], data['y']
        roc_score_value = 0
        if algo_name in (['ADERH', 'INNE', 'IForest', 'DIF']):
            for seed in random_seeds:
             algo = algo_dic[algo_name](random_state=seed)
             algo.fit(X)
             outlier_score, label = algo.decision_function(X), algo.labels_
             roc_score_value += roc_auc_score(Y, outlier_score) /len(random_seeds)
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
             algo = algo_dic[algo_name]()
             algo.fit(X)
             outlier_score, label = algo.decision_function(X), algo.labels_
             roc_score_value = roc_auc_score(Y, outlier_score)

        print('dataset:{}, algo: {}, roc-score: {} '.format(data_name.split('/')[-1].split('.')[0], algo_name,   roc_score_value))




