from sklearn import svm


def run_svc(X_train, X_test, y_train, y_test):
    # Fit to model and predict
    svc = svm.SVC()
    y_pred = svc.fit(X_train, y_train).predict(X_test)
    return y_pred
