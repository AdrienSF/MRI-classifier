import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE, SelectFromModel
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.utils import resample

# from tensorflow.keras import layers, Sequential
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, Input
# from tensorflow.keras.utils import to_categorical

from multiprocessing import Process, Queue
import mlflow

loaded_df = pd.read_csv('Volume_df.csv')
volume_df = loaded_df
# remove subject 53 an 369 (who have mostly NaN values)
volume_df = volume_df.drop([53, 369])
# remove voxel data (volume is a more universal measure)
volume_df = volume_df.drop(volume_df.columns[71:], axis='columns')

# get a matrix of data
volume_mat = volume_df.to_numpy()
volume_mat = np.array(volume_mat[:,2:], dtype=float)
# scale values to 0-1 range and seperate labels
X_norm = MinMaxScaler().fit_transform(volume_mat)
y = volume_df['Target'].to_numpy()

# decide on a reduced feature space using PCA (this can be considered double dipping as we have not bothered to leave an unseen test set)
pca = PCA(n_components=30) # previous analysis showed that the first 30 conmponents account for over 90% of variance
pca.fit(X_norm)

features_to_try = []
# get the indexes of the features that contribute to 1st pc by over .15
features_to_try.append(np.where(pca.components_[0] > .15)[0])

# get the indexes of the feature that contribute most to the first 30 pricipal components
features_to_try.append(np.array([np.argmax(pca.components_[i]) for i in range(30)]))

# get the indexes of the features that contribute to the first 15 pricipal components by over .30
indexes = []
for i in range(15):
  indexes += list(np.where(pca.components_[i] > .30)[0]) 
# remove possible duplicates
features_to_try.append(np.array(list(set(indexes))))


# compare with all features:
features_to_try.append(list(range(69)))

for selected_feature_indexes in features_to_try:

    selected_X_norm = X_norm[:, selected_feature_indexes]
    
    # log the features used as input in this experiment
    input_features = np.array(list(volume_df))[selected_feature_indexes]
    
    
    # Subsampling from the dataset 
    
    # Creating the Minority
    volume_df['Target'].value_counts()
    df_minority = volume_df[volume_df.Target=='AD']
    df_minority
    
    #Creating a Majority for CN 
    df_CN = volume_df[volume_df.Target=='CN']
    # Downsample majority class
    cn_sampled = resample(df_CN, 
                                    replace=False,    # sample without replacement
                                    n_samples=76,     # to match minority class
                                    random_state=123) # reproducible results
    
    
    #Creating a Majority for MCI 
    df_MCI = volume_df[volume_df.Target=='MCI']
    # Downsample majority class
    mci_sampled = resample(df_MCI, 
                                    replace=False,    # sample without replacement
                                    n_samples=76,     # to match minority class
                                    random_state=123) # reproducible results
    
    #Creating a Majority for MCI 
    df_SPR = volume_df[volume_df.Target=='SPR']
    # Downsample majority class
    spr_sampled = resample(df_SPR, 
                                    replace=False,    # sample without replacement
                                    n_samples=76,     # to match minority class
                                    random_state=123) # reproducible results
    
    # join resampled
    df_downsampled = pd.concat([df_minority, cn_sampled, mci_sampled, spr_sampled])
    
    
    # get a matrix of resampled data
    volume_mat = df_downsampled.to_numpy()
    volume_mat = np.array(volume_mat[:,2:], dtype=float)
    # scale values to 0-1 range and seperate labels
    X_norm = MinMaxScaler().fit_transform(volume_mat)
    y = volume_df['Target'].to_numpy()
    
    
    # set up experiment with mlflow
    mlflow.set_experiment('compare 100-fold test acc using all features vs reduced features')
    # mlflow.set_experiment('sbatch test')
    
    
    # import the class used for parallel k-fold cross-validation
    from parallel_kfold import ParallelKFold
    #Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier
    
    
    # define functions that return the accuracy of different models
    def try_adaboost(X_train, y_trian, X_val, y_val):
        model = AdaBoostClassifier(n_estimators=1000, random_state=0)
        model.fit(X_train, y_trian)
    
        mlflow.log_param('input_features', input_features)
        return model.score(X_val, y_val)
    
    def try_random_forest(X_train, y_train, X_val, y_val):
        #Create a Gaussian Classifier
        clf=RandomForestClassifier()
        #Train the model using the training sets y_pred=clf.predict(X_test)
        clf.fit(X_train, y_train)
        
        mlflow.log_param('input_features', input_features)
        return clf.score(X_val, y_val)
    
    def try_lda(X_train, y_train, X_val, y_val):
        clf = LinearDiscriminantAnalysis()
        clf.fit(X_train, y_train)
    
        mlflow.log_param('input_features', input_features)
        return clf.score(X_val, y_val)
    
    def try_svm1(X_train, y_train, X_val, y_val):
        clf = svm.SVC() # Linear Kernel
        clf.fit(X_train, y_train)
    
        mlflow.log_param('input_features', input_features)
        return clf.score(X_val, y_val)
    
    def try_svm2(X_train, y_train, X_val, y_val):
        clf = svm.SVC(kernel='poly') # Linear Kernel
        clf.fit(X_train, y_train)
    
        mlflow.log_param('input_features', input_features)
        return clf.score(X_val, y_val)
    
    def try_svm3(X_train, y_train, X_val, y_val):
        clf = svm.SVC(kernel='rbf') # Linear Kernel
        clf.fit(X_train, y_train)
    
        mlflow.log_param('input_features', input_features)
        return clf.score(X_val, y_val)
    
    def try_svm4(X_train, y_train, X_val, y_val):
        clf = svm.SVC(kernel='sigmoid') # Linear Kernel
        clf.fit(X_train, y_train)
    
        mlflow.log_param('input_features', input_features)
        return clf.score(X_val, y_val)
    
    
    models_to_try = [try_lda, try_svm1, try_svm2, try_svm3, try_svm4]
    
    pkf = ParallelKFold()
    
    # loop through functions and perform parallel cross-validation (kfold_acc is logged to mlflow by pkf.k_fold())
    mlflow.sklearn.autolog()
    for try_model in models_to_try:
        kfold_acc = pkf.k_fold(100, try_model, selected_X_norm, y)
        print(kfold_acc)