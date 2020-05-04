from sklearn.tree import DecisionTreeClassifier


def run_j48(X_train, X_test, y_train, y_test):
    # Fit to model and predict
    j48 = DecisionTreeClassifier()
    y_pred = j48.fit(X_train, y_train).predict(X_test)
    return y_pred
