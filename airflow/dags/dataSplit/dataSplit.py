import pandas as pd
from sklearn.model_selection import train_test_split
from openpyxl.workbook import Workbook
import os
import logging

def split():
    current_directory = os.path.dirname(__file__)
    # print(current_directory)
    dataSplitLog = os.path.join(current_directory, "../data/datasplit.log")

    logging.basicConfig(filename=dataSplitLog, level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logging.warning("starting new run")

    sourcePath = os.path.join(current_directory, "../data/default of credit card clients.xls")
    print(sourcePath)
    logging.info("data source picked")

    newdata = pd.read_excel(io = sourcePath, header=1, index_col=False)
    train_val_data, test_data = train_test_split(newdata, test_size=.10, random_state=42,stratify=newdata['default payment next month'])

    trainpath = os.path.join(current_directory, "../data/train_val_data.xlsx")
    testpath = os.path.join(current_directory, "../data/test_data.xlsx")

    train_val_data.to_excel(trainpath, index=False)
    logging.info("training and validation data saved")

    test_data_no_target = test_data.iloc[:,:-1] # remove target from test data
    logging.info("target removed from test data")
    
    test_data_no_target.to_excel(testpath, index=False)
    logging.info("test data saved without target")


if __name__ == "__main__":
    split()