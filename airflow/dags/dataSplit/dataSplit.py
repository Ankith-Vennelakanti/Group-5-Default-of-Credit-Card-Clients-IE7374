import pandas as pd
from sklearn.model_selection import train_test_split
from openpyxl.workbook import Workbook
import os
import logging

"""
split() reads data and splits it into 2 parts:
1. train_val_data - to train and validate the model. contains 90% of data
2. test_data - to test the prediction part of the pipeline. this data does not contain the target column. contain 10% of data

after splitting, both the data's are dumped into separate an excel files for further processing in upstream jobs.
"""

def split():
    current_directory = os.path.dirname(__file__)
    dataSplitLog = os.path.join(current_directory, "../data/datasplit.log")

    # initialize logging
    logging.basicConfig(filename=dataSplitLog, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.warning("starting new run")

    # read data
    sourcePath = os.path.join(current_directory, "../data/default of credit card clients.xls")
    logging.info("data source picked")
    newdata = pd.read_excel(io = sourcePath, header=1, index_col=False)

    # Split data
    train_val_data, test_data = train_test_split(newdata, test_size=.10, random_state=42,stratify=newdata['default payment next month'])

    trainpath = os.path.join(current_directory, "../data/train_val_data.xlsx")
    testpath = os.path.join(current_directory, "../data/test_data.xlsx")

    # dump test_val_data
    train_val_data.to_excel(trainpath, index=False)
    logging.info("training and validation data saved")

    test_data_no_target = test_data.iloc[:,:-1] # remove target from test data
    logging.info("target removed from test data")
    
    # dump test data
    test_data_no_target.to_excel(testpath, index=False)
    logging.info("test data saved without target")


if __name__ == "__main__":
    split()