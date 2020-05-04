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

    total_datapoints = X_test.shape[0]
    mislabeled_datapoints = (y_test != y_pred).sum()
    correct_datapoints = total_datapoints-mislabeled_datapoints
    percent_correct = (correct_datapoints / total_datapoints) * 100

    print("MultiLevelPerceptron Classifier results:\n")
    print("Total datapoints: %d\nCorrect datapoints: %d\nMislabeled datapoints: %d\nPercent correct: %.2f%%"
          % (total_datapoints, correct_datapoints, mislabeled_datapoints, percent_correct))
