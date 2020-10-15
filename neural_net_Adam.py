"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetworkAdam:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and performs classification
    over C classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices.

    The network uses a nonlinearity after each fully connected layer except for
    the last. The outputs of the last fully-connected layer are the scores for
    each class."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:

        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)

        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: The number of classes C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        assert len(hidden_sizes) == (num_layers - 1)
        sizes = [input_size] + hidden_sizes + [output_size]

        self.params = {}
        self.m = {}
        self.v= {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(
                sizes[i - 1], sizes[i]
            ) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros((1,sizes[i]))
            self.m[i] = np.zeros((sizes[i - 1], sizes[i]))
            self.v[i] = np.zeros((sizes[i - 1], sizes[i]))
            assert self.params["W" + str(i)].shape == self.m[i].shape
            assert self.params["W" + str(i)].shape == self.v[i].shape

        self.m_cap = 0
        self.v_cap = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.

        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias

        Returns:
            the output
        """

        return np.dot(X,W)+ b

    def relu_grad(self,X1: np.ndarray,X2)-> np.ndarray:

        """for i,k in enumerate(X):
            for j,m in enumerate(k):
                if j>0:
                    X[i,j]=1
                else:
                    X[i,j]=0
        #print(val)"""
        X1[X2 <= 0] = 0
        return X1

    def linear_grad(self,i):
        return self.params["W" + str(i)]

    def softmax_grad(self,X: np.ndarray,y: np.ndarray) -> np.ndarray:
        dscores = X
        dscores[range(len(X)), y] -= 1
        dscores /= len(X)

        return dscores

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).

        Parameters:
            X: the input data

        Returns:
            the output
        """
        return np.maximum(0,X)

    def softmax(self, scores: np.ndarray) -> np.ndarray:
        """The softmax function.

        Parameters:
            X: the input data

        Returns:
            the output
        """
        exp_scores = np.exp(scores-np.max(scores))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the scores for each class for all of the data samples.

        Hint: this function is also used for prediction.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample

        Returns:
            Matrix of shape (N, C) where scores[i, c] is the score for class
                c on input X[i] outputted from the last layer of your network
        """
        self.outputs={}
        self.linear_outputs={}
        self.linear_outputs[0]=X
        self.outputs[0] = X
        for i in range(1,self.num_layers):
            self.linear_outputs[i]=self.linear(self.params["W"+str(i)],self.outputs[i-1],self.params["b"+str(i)])
            self.outputs[i] = self.relu(self.linear_outputs[i])
        self.linear_outputs[i+1]=self.linear(self.params["W" + str(i+1)], self.outputs[i], self.params["b" + str(i+1)])
        self.outputs[i + 1] = self.softmax(self.linear_outputs[i+1])

        return self.outputs[i+1]

    def backward(self, X: np.ndarray, y: np.ndarray, epoch, lr: float, reg: float = 0.0) -> float:
        """Perform back-propagation and update the parameters using the
        gradients.

        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training sample
            y: Vector of training labels. y[i] is the label for X[i], and each
                y[i] is an integer in the range 0 <= y[i] < C
            lr: Learning rate
            reg: Regularization strength

        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        self.layer_gradients = {}
        probs=self.forward(X)
        logprobs = -np.log(probs[range(len(X)), y])
        loss = np.sum(logprobs) / len(X)
        self.grad_pass={}

        soft_grad=self.softmax_grad(probs,y)
        self.grad_pass[self.num_layers]=soft_grad
        self.gradients["W"+str(self.num_layers)]=np.dot(self.outputs[self.num_layers-1].T,soft_grad)+reg*self.params["W"+str(self.num_layers)]

        for i in range(self.num_layers-1,0,-1):
            linear_grad=np.dot(self.grad_pass[i+1],self.linear_grad(i+1).T)
            self.grad_pass[i]=self.relu_grad(linear_grad,self.linear_outputs[i])
            self.gradients["W" + str(i)] = np.dot(self.outputs[i-1].T, self.grad_pass[i])+ reg * self.params["W" + str(i)]

        for i in range(1,self.num_layers+1):

            self.m[i] = (self.beta1 * self.m[i]) + ((1 - self.beta1) * self.gradients["W" + str(i)])
            self.v[i] = (self.beta2 * self.v[i]) + (
                        (1 - self.beta2) * (self.gradients["W" + str(i)] * self.gradients["W" + str(i)]))

            self.m_cap = self.m[i]/(1 - (self.beta1 ** epoch))
            self.v_cap = self.v[i]/(1 - (self.beta2 ** epoch))

            self.params["W" + str(i)] -= (lr * self.m_cap * (1/(np.sqrt(self.v_cap) + self.epsilon)))
            self.params["b" + str(i)] += -lr * np.sum(self.grad_pass[i], axis=0, keepdims=True)

        return loss
