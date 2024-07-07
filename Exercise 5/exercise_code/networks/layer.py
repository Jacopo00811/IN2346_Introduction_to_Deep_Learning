import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return out: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        shape = x.shape
        out, cache = np.zeros(shape), np.zeros(shape)
        ########################################################################
        # TODO:                                                                #
        # Implement the forward pass of Sigmoid activation function            #
        ########################################################################

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        out = 1/(1+np.exp(-x))
        cache = out

        
        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropogation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None

        out = cache
        cache = None

        dx = dout * out * (1 - out)

        return dx


class Relu:
    def __init__(self):
        pass

    def forward(self, x):
        """
        :param x: Inputs, of any shape.

        :return outputs: Outputs, of the same shape as x.
        :return cache: Cache, stored for backward computation, of the same shape as x.
        """
        out = None
        cache = None
        
        out = np.maximum(0,x)
        cache = out

        return out, cache

    def backward(self, dout, cache):
        """
        :param dout: Upstream gradient from the computational graph, from the Loss function
                    and up to this layer. Has the shape of the output of forward().
        :param cache: The values that were stored during forward() to the memory,
                    to be used during the backpropogation.
        :return: dx: the gradient w.r.t. input X, of the same shape as X
        """
        dx = None

        out = cache
        cache = None

        dx = dout * (out > 0) # out > 0 --> 1, out < 0 --> 0

        return dx


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.
    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.
    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M, 1)
    :return out: output, of shape (N, M)
    :return cache: (x, w, b)
    """
    N, M = x.shape[0], b.shape[0]
    out = np.zeros((N,M))

    # # Example input
    # x = np.array([[[1, 2, 3], [4, 5, 6]],
    #           [[7, 8, 9], [10, 11, 12]],
    #           [[13, 14, 15], [16, 17, 18]]])
    
    # N = 3, d_1(rows) = 2, d_2(colums) = 3
    # x.shape = (3, 2, 3)
    # call the x.reshape(N, -1) function to flatten the input
    # Flattened x:
    # [[ 1  2  3  4  5  6]
    # [ 7  8  9 10 11 12]
    # [13 14 15 16 17 18]]
    # x.shape = (N, D) = (3, 6) where D = d_1 * d_2 = 2 * 3 = 6


    out = np.dot(x.reshape(N,-1), w) + b # (N, D) * (D, M) + (M, 1) = (N, M)


    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    Inputs:
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M, 1)
    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache # b actually not needed since the derivative of z = XW + b with respect to b is 1
    dx, dw, db = None, None, None
    
    dx = np.dot(dout, w.T).reshape(x.shape) # (N, M) * (M, D) = (N, D) = (N, d_1, ..., d_k)
    # x.reshape(x.shape[0], -1) --> (N, D)
    dw = np.dot(x.reshape(x.shape[0], -1).T, dout) # (D, N) * (N, M) = (D, M)
    db = np.sum(dout, axis=0) # (N, M) --> (M, 1)

    return dx, dw, db