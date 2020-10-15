"""Perceptron model."""

import numpy as np


class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """

        self.w = []
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Use the perceptron update rule as introduced in Lecture 3.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        self.w = np.random.rand(self.n_class, np.shape(X_train)[1])

        for i in range(self.epochs):
            for l in range(len(X_train)):
                for k in range(self.n_class):
                    if np.dot(self.w[k],X_train[l]) > np.dot(self.w[y_train[l]], X_train[l]):
                        self.w[y_train[l]] = self.w[y_train[l]] + (self.lr * X_train[l])
                        self.w[k] = self.w[k] - (self.lr * X_train[l])

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
            y_pred = np.dot(self.w, X_test[i])
            predictions[i] = np.argmax(y_pred)

        return predictions
