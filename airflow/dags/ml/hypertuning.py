import os
import pickle
import lightgbm as lgb
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

def tune():

    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    traindata = os.path.join(current_directory, "data/train_val_data.pkl")
    testdata = os.path.join(current_directory, "data/test_val_data.pkl")

     # load train data
    with open(traindata, 'rb') as file:
        train = pickle.load(file)

    # load test data
    with open(testdata, 'rb') as file:
        test = pickle.load(file)

    # logging.info("data loaded")

    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    scaler = StandardScaler()
    scaler.fit(X_train)

    scaled_train = scaler.transform(X_train)
    scaled_test = scaler.transform(X_test)

    model = lgb.LGBMClassifier(metric='auc')

    lgb_params = {'learning_rate': [0.01, 0.001,0.05,0.5],
                'max_depth': [2, 3, 4, 5, 8, 10]}

    grid_search = GridSearchCV(model, lgb_params, cv=5, scoring='accuracy')

    grid_search.fit(scaled_train, y_train)

    # print the best parameters and score
    print("Best parameters: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)

if __name__ == "__main__":
    tune()