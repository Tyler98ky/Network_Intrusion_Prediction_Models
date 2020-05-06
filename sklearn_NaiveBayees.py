from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt


def run_naive_bayes(X_train, X_test, y_train, y_test):
    # Fit to model and predict
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    gnb_disp = plot_confusion_matrix(gnb, X_test, y_test)
    # plt.show()
    return y_pred
