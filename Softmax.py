"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """

        g_list = [0]*self.n_class

        for i in range(len(X_train)):
            exp_list = np.exp(np.dot(self.w, X_train[i])-np.max(np.dot(self.w, X_train[i])))
            for j in range(self.n_class):
                if j == y_train[i]:
                        dw = ((exp_list[j] * X_train[i]) / exp_list.sum()) - X_train[i]
                        g_list[j] = g_list[j] + dw
                else:
                        dw = (exp_list[j] * X_train[i])/exp_list.sum()
                        g_list[j] = g_list[j] + dw

        return g_list

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """

        self.w = np.random.rand(self.n_class, np.shape(X_train)[1])
        mini_batch_size = 200

        for i in range(self.epochs):
            for j in range(0, len(X_train), mini_batch_size):
                if (j + mini_batch_size) < len(X_train):
                    x_train_mini = X_train[j:j+mini_batch_size]
                    y_train_mini = y_train[j:j+mini_batch_size]
                    gradient = self.calc_gradient(x_train_mini, y_train_mini)
                    for k,l in enumerate(gradient):
                        self.w[k] = self.w[k] - (self.lr * l)

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

        predictions = np.zeros(len(X_test))

        for i in range(len(X_test)):
            exp_list = np.exp(np.dot(self.w, X_test[i])-np.max(np.dot(self.w, X_test[i])))
            predictions[i] = np.argmax(exp_list/exp_list.sum())

        return predictions
