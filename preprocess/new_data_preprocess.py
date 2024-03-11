from preprocess import pre_process
import os
import pickle

def main():
    pre = pre_process()

    current_directory = os.path.dirname(__file__)
    testpath = os.path.join(current_directory, "../data/test_data.xlsx")

    data = pre.process(testpath)
    data = pre.remove_target(data)

    with open('new_processed_data.pkl', 'wb') as file:
        pickle.dump(data, file)

if __name__ == "__main__":
    main()
