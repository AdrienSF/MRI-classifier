import shap
import mlflow
import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier



def resample_balanced(volume_filename: str):
    loaded_df = pd.read_csv(volume_filename)
    volume_df = loaded_df
    # remove subject 53 an 369 (who have mostly NaN values)
    volume_df = volume_df.drop([53, 369])
    # remove voxel data (volume is a more universal measure)
    volume_df = volume_df.drop(volume_df.columns[71:], axis='columns')
    # remove subject id
    volume_df = volume_df.drop(volume_df.columns[0], axis='columns')

    # add other label format
    volumes["Target"] = volume_df["Target"].astype('category')
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




def save_shap_plots(clf, data, labels, plot_labels):
    clf_name = type(clf)
    # train on all data, we are not interested in test accuracy but rather which features the classifier finds important.
    # Thus we should train it on as much data as we can in order to more accurately meaure feature salience 
    rforest.fit(X_train, y_train)
    # explain all the predictions in the test set
    rfexplainer = shap.KernelExplainer(rforest.predict_proba, X_train)
    rfshap_values = rfexplainer.shap_values(X_test)
    # save plot of overall salience
    shap.summary_plot(rfshap_values, X_test, class_names=df_downsampled['Target'], show=False)
    plt.savefig(clf_name+'mean(|SHAP_val|)')
    # save plot of impact on output 
    shap.summary_plot(rfshap_values[0], X_test, class_names=df_downsampled['Target'])
    plt.savefig(clf_name+'SHAP_val')