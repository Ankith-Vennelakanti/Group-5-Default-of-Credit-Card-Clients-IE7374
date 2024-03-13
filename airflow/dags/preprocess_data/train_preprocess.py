from preprocess_data.preprocess import process
import os
import pickle

def train_process():
    # pre = pre_process()

    # current_directory = os.path.dirname(__file__)
    current_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    trainpath = os.path.join(current_directory, "data/train_val_data.xlsx")

    print(trainpath)

    data = process(path = trainpath)
   
    with open('processed_data.pkl', 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    train_process()