from preprocess_data.preprocess import process,remove_target
import os
import pickle

def new_data():
    # pre = pre_process()

    current_directory = os.path.dirname(__file__)
    testpath = os.path.join(current_directory, "../data/test_data.xlsx")

    data = process(testpath)
    data = remove_target(data)

    with open('new_processed_data.pkl', 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    new_data()
