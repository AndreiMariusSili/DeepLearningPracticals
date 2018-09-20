"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
from typing import Dict
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    _in: np.ndarray
    _out: np.ndarray
    params: Dict[str, np.ndarray]
    grads: Dict[str, np.ndarray]

    def __init__(self, in_features: int, out_features: int):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample

        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.

        Also, initialize gradients with zeros.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.in_dim = in_features
        self.out_dim = out_features

        self.params = {
            'weight': np.random.normal(loc=0.0, scale=0.0001, size=[out_features, in_features]),
            'bias': np.zeros([out_features, 1])
        }
        self.grads = {
            'weight': np.zeros([out_features, in_features]),
            'bias': np.zeros([out_features, 1])
        }
        ########################
        # END OF YOUR CODE    #
        #######################

    def __repr__(self):
        return "Linear: {} -> {}".format(self.in_dim, self.out_dim)

    def __str__(self):
        return "Linear: {} -> {}".format(self.in_dim, self.out_dim)

    def forward(self, x: np.ndarray):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        out = x @ self.params['weight'].T + self.params['bias'].T

        self._in = x.copy()
        self._out = out.copy()

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout: np.ndarray):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.grads['bias'] = dout.sum(axis=0).reshape(-1, 1)
        self.grads['weight'] = dout.T @ self._in
        dx = dout @ self.params['weight']

        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class ReLUModule(object):
    """
    ReLU activation module.
    """
    _in: np.ndarray
    _out: np.ndarray

    def __repr__(self):
        return "ReLU"

    def __str__(self):
        return "ReLU"

    def forward(self, x: np.ndarray):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        out = x.copy()
        out[x < 0] = 0

        self._in = x.copy()
        self._out = out.copy()

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout: np.ndarray):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
          math_correct: whether to use the mathematically sound implementation or a trick to compute the derivative
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        # almost mathematically sound
        # ind = np.zeros_like(self._in)
        # ind[self._in > 0] = 1
        # diag = np.einsum('ij,mj -> imj', ind, np.eye(self._out.shape[1]))
        # dx = np.einsum('ij,ijp->ip', dout, diag)

        dx = dout.copy()
        dx[self._in < 0] = 0
        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def __repr__(self):
        return "SoftMax"

    def __str__(self):
        return "SoftMax"

    _in: np.ndarray
    _out: np.ndarray

    def forward(self, x: np.ndarray):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        b: np.ndarray = x.max(axis=1).reshape(-1, 1)
        y: np.ndarray = np.exp(x - b)
        out = y / y.sum(axis=1).reshape(-1, 1)

        self._in = x.copy()
        self._out = out.copy()

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout: np.ndarray):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        diag = np.einsum('ij,mj -> imj', self._out, np.eye(self._out.shape[1]))
        outer = np.einsum('ij,ik->ijk', self._out, self._out)
        dx = np.einsum('ij,ijp->ip', dout, (diag - outer))

        ########################
        # END OF YOUR CODE    #
        #######################

        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x: np.ndarray, y: np.ndarray):
        """
        Forward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        batch_truth = np.argmax(y, axis=1)
        batch_truth_probs = x[np.arange(x.shape[0]), batch_truth]
        batch_size = x.shape[0]
        out = 1/batch_size * np.sum(-np.log(batch_truth_probs))

        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x: np.ndarray, y: np.ndarray):
        """
        Backward pass.

        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        batch_size = x.shape[0]
        dx = (-1/batch_size * y / x)

        ########################
        # END OF YOUR CODE    #
        #######################

        return dx
