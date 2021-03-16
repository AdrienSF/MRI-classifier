#!/usr/bin/python3

#SBATCH --nodes=1

#SBATCH --time=0-10:00:00
#SBATCH --job-name=MLflow
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import LeaveOneOut

from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, Input
from tensorflow.keras.utils import to_categorical

from multiprocessing import Process, Queue
import mlflow
mlflow.sklearn.autolog()


loaded_df = pd.read_csv('Volume_df.csv')
volume_df = loaded_df
# remove subject 53 an 369 (who have mostly NaN values)
volume_df = volume_df.drop([53, 369])
# get a matrix of data
volume_mat = volume_df.to_numpy()
volume_mat = np.array(volume_mat[:,2:], dtype=float)
# remove voxel data (volume is a more universal measure)
volume_mat = volume_mat[:,:69]

X_norm = MinMaxScaler().fit_transform(volume_mat)



###################### classifier attempt
y = volume_df['Target'].to_numpy()
X_norm = MinMaxScaler().fit_transform(volume_mat)

def try_model(X_train, y_trian, X_val, y_val, res_queue):
    model = AdaBoostClassifier(n_estimators=10, random_state=0)
    model.fit(X_train, y_trian)

    res_queue.put(model.score(X_val, y_val))


res_queue = Queue()
loo = LeaveOneOut()
looprocs = []

with mlflow.start_run() as run:

    for train_index, test_index in loo.split(X_norm):
        p = Process(target=try_model, args=(X_norm[train_index], y[train_index], X_norm[test_index], y[test_index], res_queue))
        p.start()
        looprocs.append(p)
        
    for proc in looprocs:
        proc.join()

    mlflow.log_metric('res_queue', str(res_queue))