

#KDD2024
from ARDEH import ADERH
import numpy as np
path_to_file = ''
data = np.load('{}'.format(path_to_file),
               allow_pickle=True)
X, Y = data['X'], data['y']

aderh = ADERH()
aderh.fit(X)
outlier_scoer, label = aderh.decision_function(X), aderh.labels_
