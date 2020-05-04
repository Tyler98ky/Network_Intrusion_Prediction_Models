from sklearn import preprocessing
from scipy.io.arff import loadarff
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

def run_random_forests_classifier(X_train, X_test, y_train, y_test):
    # Fit to model and predict
    rfc = RandomForestClassifier()
    y_pred = rfc.fit(X_train, y_train).predict(X_test)

    total_datapoints = X_test.shape[0]
    mislabeled_datapoints = (y_test != y_pred).sum()
    correct_datapoints = total_datapoints-mislabeled_datapoints
    percent_correct = (correct_datapoints / total_datapoints) * 100

    print("RandomForestClassifier results:\n")
    print("Total datapoints: %d\nCorrect datapoints: %d\nMislabeled datapoints: %d\nPercent correct: %.2f%%"
          % (total_datapoints, correct_datapoints, mislabeled_datapoints, percent_correct))
