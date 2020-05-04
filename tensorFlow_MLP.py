from keras.models import Sequential
from keras.layers import Dense, Dropout


def run_mlp(X_train, X_test, y_train, y_test):
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    model.fit(X_train, y_train,
              epochs=500,
              batch_size=128)
    score = model.evaluate(X_test, y_test, batch_size=128)

    total_datapoints = X_test.shape[0]
    percent_correct = score[1] * 100
    correct_datapoints = int(round(total_datapoints * percent_correct) / 100)
    mislabeled_datapoints = total_datapoints - correct_datapoints


    print("MultiLevelPerceptron Classifier results for NSL-KDD using TensorFlow and Keras:\n")
    print("Total datapoints: %d\nCorrect datapoints: %d\nMislabeled datapoints: %d\nPercent correct: %.2f%%"
          % (total_datapoints, correct_datapoints, mislabeled_datapoints, percent_correct))