import pathlib
import shap
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from parallel_kfold import ParallelKFold


def resample_balanced(volume_filename: str):
    volumes = pd.read_csv(volume_filename)

    # add other label format
    volumes["Target"] = volumes["Target"].astype('category')
    volumes['Target_cat'] = volumes['Target'].cat.codes 


    # Subsampling from the dataset 
    # Creating the Minority
    volumes['Target'].value_counts()
    df_minority = volumes[volumes.Target=='AD']
    df_minority

    #Creating a Majority for CN 
    df_CN = volumes[volumes.Target=='CN']
    # Downsample majority class
    cn_sampled = resample(df_CN, 
                                    replace=False,    # sample without replacement
                                    n_samples=76,     # to match minority class
                                    random_state=123) # reproducible results


    #Creating a Majority for MCI 
    df_MCI = volumes[volumes.Target=='MCI']
    # Downsample majority class
    mci_sampled = resample(df_MCI, 
                                    replace=False,    # sample without replacement
                                    n_samples=76,     # to match minority class
                                    random_state=123) # reproducible results

    #Creating a Majority for MCI 
    df_SPR = volumes[volumes.Target=='SPR']
    # Downsample majority class
    spr_sampled = resample(df_SPR, 
                                    replace=False,    # sample without replacement
                                    n_samples=76,     # to match minority class
                                    random_state=123) # reproducible results

    

    # creating a new df with subsampled dfs
    pdList = [cn_sampled, mci_sampled, spr_sampled]  # List of your dataframes
    df_majority = pd.concat(pdList)
    df_majority["Target"].value_counts()


    # Combine minority class with downsampled majority class
    df_downsampled = pd.concat([df_minority, df_majority])


    limited = df_downsampled
    limited = limited.drop(columns=['Target', 'Target_cat'])


    return limited, df_downsampled


def run_svm(X_train, y_train, X_val, y_val):
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    mlflow.log_param('input_features', input_features)
    return clf.score(X_val, y_val)




limited, full = resample_balanced('volumes.csv')

loocv = ParallelKFold()
acc = loocv.k_fold(limited.shape[0]/2, run_svm, limited.to_numpy(), full['Target_cat'].to_numpy())
print("linear svm leave one out cross-validation accuracy (balanced unscaled dataset):", acc)



limited, full = resample_balanced('scaled_volumes.csv')

loocv = ParallelKFold()
acc = loocv.k_fold(limited.shape[0]/2, run_svm, limited.to_numpy(), full['Target_cat'].to_numpy())
print("linear svm leave one out cross-validation accuracy (balanced unscaled dataset):", acc)