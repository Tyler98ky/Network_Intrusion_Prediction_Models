from sklearn.ensemble import RandomForestClassifier


def run_random_forests_classifier(X_train, X_test, y_train, y_test):
    # Fit to model and predict
    rfc = RandomForestClassifier()
    y_pred = rfc.fit(X_train, y_train).predict(X_test)
    return y_pred
