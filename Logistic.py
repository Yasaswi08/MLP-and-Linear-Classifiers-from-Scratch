"""Logistic regression model."""

import numpy as np


class Logistic:
    def __init__(self, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.threshold = 0.5

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid function.

        Parameters:
            z: the input

        Returns:
            the sigmoid of the input
        """

        sig_res = 1 / (1 + np.exp((-1) * z))

        return sig_res

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the logistic regression update rule as introduced in lecture.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        self.w = np.random.rand(1, np.shape(X_train)[1])

        for l in range(len(y_train)):
            if y_train[l] == 0:
                y_train[l] = -1

        for j in range(self.epochs):
            inter_res = 0
            for i in range(len(X_train)):
                inter_res = inter_res + self.sigmoid((np.dot(self.w, X_train[i]) * (y_train[i])) * (-1)) * (
                X_train[i]) * \
                            (y_train[i])
            dw = (-1) * inter_res
            self.w = self.w - (self.lr * dw)

        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """

        predictions = np.zeros((len(X_test)))

        for i in range(len(X_test)):
            predictions[i] = self.sigmoid(np.dot(self.w, X_test[i]))
        for j, k in enumerate(predictions):
            if k > 0.5:
                predictions[j] = 1
            else:
                predictions[j] = -1

        return predictions
