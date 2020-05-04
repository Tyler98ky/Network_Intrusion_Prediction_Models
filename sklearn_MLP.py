from sklearn.neural_network import MLPClassifier


def run_mlp(X_train, X_test, y_train, y_test):

    # Weka arguments converted to scikit-learn
    # L -> learning_rate_init
    # H -> hidden_layer_size
    # M -> momentum
    # N -> max_iter
    # V -> validation_fraction
    # S -> random_state

    # Fit to model and predict
    clf = MLPClassifier(solver='sgd', alpha=1e-5,
                        hidden_layer_sizes=(22,), random_state=0, learning_rate_init=.3,
                        momentum=.2, max_iter=500, validation_fraction=0, )
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    return y_pred

