import pandas as pd
import os
import pickle
import logging

"""
    Prepare and process data for retraining.

    This function performs the following steps:
    1. Loads data from pickle file 'prpkl.pkl'.
    2. Loads additional training data from Excel file 'train_val_data.xlsx'.
    3. Renames the 'preds' column in the pickle data to 'default payment next month'.
    4. Concatenates the loaded training data with the pickle data.
    5. Drops unnecessary columns from the concatenated data.
    6. Saves the processed data to a pickle file 'train_processed_data.pkl'.

    Returns:
    None
"""


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