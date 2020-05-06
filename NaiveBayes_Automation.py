import Network_Intrusion_Prediction_Models
import sklearn_NaiveBayees


def main():
    train, test = Network_Intrusion_Prediction_Models.load_nb15_csv()
    X_train, X_test, y_train, y_test = Network_Intrusion_Prediction_Models.preprocess_data(train, test)

    for i in range(10):
        prediction_results = sklearn_NaiveBayees.run_naive_bayes(X_train, X_test, y_train, y_test)
        Network_Intrusion_Prediction_Models.log_results(prediction_results, y_test)


if __name__ == '__main__':
    main()