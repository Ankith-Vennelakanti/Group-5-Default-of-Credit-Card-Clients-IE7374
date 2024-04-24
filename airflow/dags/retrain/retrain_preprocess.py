import pandas as pd
import os
import pickle
import logging

def data_prep():
    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prpkl = os.path.join(current_directory, "data/prpkl.pkl")
    df_from_pickle = pd.read_pickle(prpkl)

    sourcePath = os.path.join(current_directory, "data/train_val_data.xlsx")
    logging.info("data source picked")
    traindata = pd.read_excel(io = sourcePath, index_col=False)

    df_from_pickle.rename(columns={'preds': 'default payment next month'}, inplace=True)

    retrain_data = pd.concat([traindata, df_from_pickle])

    retrain_data = retrain_data.drop(columns=['ID','EDUCATION', 'MARRIAGE','AGE'])
        
    retrainpkl = os.path.join(current_directory, "data/train_processed_data.pkl")
    retrain_data.to_pickle(retrainpkl)