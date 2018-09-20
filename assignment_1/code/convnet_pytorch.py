"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict

from torch import nn


class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """

    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem


        TODO:
        Implement initialization of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################
        super(ConvNet, self).__init__()
        self.model = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(n_channels, 64, 3, 1, 1)),
            ('conv1_bn', nn.BatchNorm2d(64)),
            ('conv1_r', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(3, 2, 1)),
            ('conv2', nn.Conv2d(64, 128, 3, 1, 1)),
            ('conv2_bn', nn.BatchNorm2d(128)),
            ('conv2_r', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(3, 2, 1)),
            ('conv3_a', nn.Conv2d(128, 256, 3, 1, 1)),
            ('conv3_a_bn', nn.BatchNorm2d(256)),
            ('conv3_a_r', nn.ReLU()),
            ('conv3_b', nn.Conv2d(256, 256, 3, 1, 1)),
            ('conv3_b_bn', nn.BatchNorm2d(256)),
            ('conv3_b_r', nn.ReLU()),
            ('maxpool3', nn.MaxPool2d(3, 2, 1)),
            ('conv4_a', nn.Conv2d(256, 512, 3, 1, 1)),
            ('conv4_a_bn', nn.BatchNorm2d(512)),
            ('conv4_a_r', nn.ReLU()),
            ('conv4_b', nn.Conv2d(512, 512, 3, 1, 1)),
            ('conv4_b_bn', nn.BatchNorm2d(512)),
            ('conv4_b_r', nn.ReLU()),
            ('maxpool4', nn.MaxPool2d(3, 2, 1)),
            ('conv5_a', nn.Conv2d(512, 512, 3, 1, 1)),
            ('conv5_a_bn', nn.BatchNorm2d(512)),
            ('conv5_a_r', nn.ReLU()),
            ('conv5_b', nn.Conv2d(512, 512, 3, 1, 1)),
            ('conv5_b_bn', nn.BatchNorm2d(512)),
            ('conv5_b_r', nn.ReLU()),
            ('maxpool5', nn.MaxPool2d(3, 2, 1)),
            ('avgpool', nn.AvgPool2d(1, 1, 0)),
        ]))
        self.linear = nn.Linear(512, n_classes)

        ########################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """

        ########################
        # PUT YOUR CODE HERE  #
        #######################

        intermediate = self.model(x)
        out = self.linear(intermediate.reshape(intermediate.shape[0], -1))

        ########################
        # END OF YOUR CODE    #
        #######################

        return out
