import pandas as pd
from sklearn.model_selection import train_test_split
from openpyxl.workbook import Workbook
import os

def split():
    current_directory = os.path.dirname(__file__)
    sourcePath = os.path.join(current_directory, "../data/default of credit card clients.xls")
    print(sourcePath)

    newdata = pd.read_excel(io = sourcePath, header=1, index_col=False)
    train_val_data, test_data = train_test_split(newdata, test_size=.10, random_state=42,stratify=newdata['default payment next month'])

    trainpath = os.path.join(current_directory, "../data/train_val_data.xlsx")
    testpath = os.path.join(current_directory, "../data/test_data.xlsx")

    train_val_data.to_excel(trainpath, index=False)
    test_data.to_excel(testpath, index=False)