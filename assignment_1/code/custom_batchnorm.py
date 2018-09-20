import numpy as np
import torch
import torch.nn as nn

"""
The modules/function here implement custom versions of batch normalization in PyTorch.
In contrast to more advanced implementations no use of a running mean/variance is made.
You should fill in code into indicated sections.
"""


######################################################################################
# Code for Question 3.1
######################################################################################

class CustomBatchNormAutograd(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    The operations called in self.forward track the history if the input tensors have the
    flag requires_grad set to True. The backward pass does not need to be implemented, it
    is dealt with by the automatic differentiation provided by PyTorch.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormAutograd object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability

        TODO:
          Save parameters for the number of neurons and eps.
          Initialize parameters gamma and beta via nn.Parameter
        """
        super(CustomBatchNormAutograd, self).__init__()

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        self.dim = n_neurons
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.dim, dtype=torch.float))
        self.beta = nn.Parameter(torch.zeros(self.dim, dtype=torch.float))

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, inp: torch.Tensor):
        """
        Compute the batch normalization

        Args:
          inp: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor

        TODO:
          Check for the correctness of the shape of the input tensor.
          Implement batch normalization forward pass as given in the assignment.
          For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        if inp.shape[1] != self.dim:
            raise ValueError('Input tensor shape {} is does not match layer shape {}'.format(inp.shape[1], self.dim))
        batch_size = inp.shape[0]
        mu = 1/batch_size * inp.sum(dim=0)
        var = 1/batch_size * ((inp-mu) ** 2).sum(dim=0)
        norm_imp = (inp - mu) / (var + self.eps).sqrt()
        out = self.gamma * norm_imp + self.beta
        ########################
        # END OF YOUR CODE    #
        #######################

        return out


######################################################################################
# Code for Question 3.2 b)
######################################################################################


class CustomBatchNormManualFunction(torch.autograd.Function):
    """
    This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
    Using torch.autograd.Function allows you to write a custom backward function.
    The function will be called from the nn.Module CustomBatchNormManualModule
    Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
    pass is done via the backward method.
    The forward pass is not called directly but via the apply() method. This makes sure that the context objects
    are dealt with correctly. Example:
      my_bn_fct = CustomBatchNormManualFunction()
      normalized = fct.apply(input, gamma, beta, eps)
    """

    @staticmethod
    def forward(ctx, inp, gamma, beta, eps=1e-5):
        """
        Compute the batch normalization

        Args:
          ctx: context object handling storing and retrival of tensors and constants and specifying
               whether tensors need gradients in backward pass
          inp: input tensor of shape (n_batch, n_neurons)
          gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
          beta: mean bias tensor, applied per neuron, shpae (n_neurons)
          eps: small float added to the variance for stability
        Returns:
          out: batch-normalized tensor

        TODO:
          Implement the forward pass of batch normalization
          Store constant non-tensor objects via ctx.constant=myconstant
          Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
          Intermediate results can be decided to be either recomputed in the backward pass or to be stored
          for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
          For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        batch_size = inp.shape[0]
        mu = 1/batch_size * inp.sum(dim=0)
        var = 1/batch_size * ((inp-mu) ** 2).sum(dim=0)
        prec = 1 / (var + eps).sqrt()
        norm_in = (inp - mu) * prec
        out = gamma * norm_in + beta

        ctx.save_for_backward(prec, norm_in, gamma)
        ctx.eps = eps
        ########################
        # END OF YOUR CODE    #
        #######################

        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        Compute backward pass of the batch normalization.

        Args:
          ctx: context object handling storing and retrieval of tensors and constants and specifying
               whether tensors need gradients in backward pass
        Returns:
          out: tuple containing gradients for all input arguments

        TODO:
          Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
          Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
          inputs to None. This should be decided dynamically.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        B, D = grad_output.shape
        prec, norm_in, gamma = ctx.saved_tensors

        grad_norm_in = grad_output * gamma
        grad_input = (1. / B) * prec * (B * grad_norm_in - torch.sum(grad_norm_in, dim=0) -
                                        norm_in * torch.sum(grad_norm_in * norm_in, dim=0))
        grad_gamma = torch.sum(norm_in * grad_output, dim=0)
        grad_beta = torch.sum(grad_output, dim=0)

        ########################
        # END OF YOUR CODE    #
        #######################

        # return gradients of the three tensor inputs and None for the constant eps
        return grad_input, grad_gamma, grad_beta, None


######################################################################################
# Code for Question 3.2 c)
######################################################################################

class CustomBatchNormManualModule(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    In self.forward the functional version CustomBatchNormManualFunction.forward is called.
    The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
    """

    def __init__(self, n_neurons, eps=1e-5):
        """
        Initializes CustomBatchNormManualModule object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability

        TODO:
          Save parameters for the number of neurons and eps.
          Initialize parameters gamma and beta via nn.Parameter
        """
        super(CustomBatchNormManualModule, self).__init__()

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        self.dim = n_neurons
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(self.dim, dtype=torch.float))
        self.beta = nn.Parameter(torch.zeros(self.dim, dtype=torch.float))
        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, inp):
        """
        Compute the batch normalization via CustomBatchNormManualFunction

        Args:
          inp: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor

        TODO:
          Check for the correctness of the shape of the input tensor.
          Instantiate a CustomBatchNormManualFunction.
          Call it via its .apply() method.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        if inp.shape[1] != self.dim:
            raise ValueError('Input tensor dimension {} does not match layer dimension {}'.format(inp.shape[1], self.dim))
        bn = CustomBatchNormManualFunction()
        out = bn.apply(inp, self.gamma, self.beta, self.eps)
        ########################
        # END OF YOUR CODE    #
        #######################

        return out
