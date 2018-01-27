import numpy as np
from loglinear import softmax


def classifier_output(x, params):
    """
    Return the output layer (class probabilities)
    of a log-linear classifier with given params on input x.
    """
    W, b = params
    W_t = W.transpose()
    mult = np.dot(W_t, x)
    grades = mult + b
    probs = softmax(grades)
    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    Compute the loss and the gradients at point x with given parameters.
    y is a scalar indicating the correct label.

    returns:
        loss,[gW,gb]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    """
    W, b = params
    # YOU CODE HERE
    y_hat = classifier_output(x, params)
    loss = -np.log(y_hat[y])
    y_hat[y] -= 1
    gW = np.outer(x, y_hat)
    gb = y_hat
    return loss, [gW, gb]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.
    """
    params = []
    return params

