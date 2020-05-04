from sklearn.naive_bayes import GaussianNB
import sklearn as sk

def run_naive_bayes(X_train, X_test, y_train, y_test):
    # Fit to model and predict
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)

    positve_label = "anomaly"
    negative_label = "normal"
    padding = " " * max(len(positve_label), len(negative_label))

    print("")
    print("{}{} {}".format(padding, positve_label, negative_label))
    confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("{} {}".format(positve_label, confusion_matrix[0]))
    print("{} {}".format(negative_label, confusion_matrix[1]))
    print("Percent correct: {:.3%}".format(sk.metrics.accuracy_score(y_test, y_pred)))
