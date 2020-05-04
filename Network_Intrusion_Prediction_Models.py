import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.io.arff import loadarff

# Algorithms we implemented
import sklearn_NaiveBayees
import tensorFlow_MLP
import sklearn_SVC
import sklearn_RandomForestsClassifier


def load_nb15_csv():
    nb15_train = pd.read_csv('Datasets/UNSW-NB15/csv/UNSW_NB15_training-set.csv', delimiter=',')
    nb15_test = pd.read_csv('Datasets/UNSW-NB15/csv/UNSW_NB15_testing-set.csv', delimiter=',')
    return nb15_train.to_numpy(), nb15_test.to_numpy()


def load_nslkdd_arff():
    kdd_train, train_metadata = loadarff("Datasets/NSL-KDD/arff/KDDTrain+.arff")
    kdd_test, test_metadata = loadarff("Datasets/NSL-KDD/arff/KDDTest+.arff")
    return np.asarray(kdd_train.tolist()), np.asarray(kdd_test.tolist())


datasets = {1: ("UNSW-NB15 (csv)", load_nb15_csv),
            2: ("NSL-KDD (arff)", load_nslkdd_arff)}

algorithms = {1: ("Naive Bayes", sklearn_NaiveBayees.run_naive_bayes),
              2: ("MLP", tensorFlow_MLP.run_mlp),
              3: ("SVC", sklearn_SVC.run_svc),
              4: ("Random Forests Classifier", sklearn_RandomForestsClassifier.run_random_forests_classifier)}


def main():
    # Receive user input for which dataset to utilize and then preprocess to prepare for algorithms
    dataset_selection = get_dataset_selection()
    print("You selected {}, beginning to preprocess data...\n\n".format(datasets[dataset_selection][0]))
    train, test = datasets[dataset_selection][1]()
    X_train, X_test, y_train, y_test = preprocess_data(train, test)

    # Determine which algorithm to use then execute
    algorithm_selection = get_algorithm_selection()
    algorithms[algorithm_selection][1](X_train, X_test, y_train, y_test)


def preprocess_data(training_nparray, testing_nparray):
    # Preprocess
    enc = preprocessing.OrdinalEncoder()

    encoded_dataset = enc.fit_transform(training_nparray)  # All categorical features are now numerical
    X_train = encoded_dataset[:, :-1]  # All rows, omit last column
    y_train = np.ravel(encoded_dataset[:, -1:])  # All rows, only the last column

    # Repeat preprocessing for test data
    encoded_dataset = enc.fit_transform(testing_nparray)
    X_test = encoded_dataset[:, :-1]
    y_test = np.ravel(encoded_dataset[:, -1:])
    return X_train, X_test, y_train, y_test


def display_list(data):
    while True:
        print("Which dataset would you like to use:")
        for i in range(len(data)):
            print("{}. {}".format(i + 1, data[i + 1][0]))

        try:
            user_input = int(input("Your selection: "))
            if user_input > len(data) or user_input < 1:
                raise ValueError
        except ValueError:
            print("\nPlease enter a valid number\n\n")
            continue
        else:
            return user_input


def get_dataset_selection():
    return display_list(datasets)


def get_algorithm_selection():
    return display_list(algorithms)


if __name__ == "__main__":
    main()