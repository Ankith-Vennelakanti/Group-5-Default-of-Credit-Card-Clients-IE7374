from preprocess import pre_process
import os
import pickle

def main():
    pre = pre_process()

    current_directory = os.path.dirname(__file__)
    trainpath = os.path.join(current_directory, "../data/train_val_data.xlsx")

    data = pre.process(trainpath)
   
    with open('processed_data.pkl', 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    main()
