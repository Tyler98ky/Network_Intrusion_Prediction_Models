from sklearn import svm


def run_svc(X_train, X_test, y_train, y_test):
    # Fit to model and predict
    svc = svm.SVC()
    y_pred = svc.fit(X_train, y_train).predict(X_test)

    total_datapoints = X_test.shape[0]
    mislabeled_datapoints = (y_test != y_pred).sum()
    correct_datapoints = total_datapoints-mislabeled_datapoints
    percent_correct = (correct_datapoints / total_datapoints) * 100

    print("SupportVectorClassification results:\n")
    print("Total datapoints: %d\nCorrect datapoints: %d\nMislabeled datapoints: %d\nPercent correct: %.2f%%"
          % (total_datapoints, correct_datapoints, mislabeled_datapoints, percent_correct))
