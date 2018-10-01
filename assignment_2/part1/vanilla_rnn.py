################################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


################################################################################

class VanillaRNN(nn.Module):

    def __init__(self, seq_length: int, input_dim: int, num_hidden: int, num_classes: int, batch_size: int,
                 device):
        super(VanillaRNN, self).__init__()

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        self.W_hx = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(input_dim, num_hidden)))
        self.W_hh = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_hidden, num_hidden)))
        self.b_h = nn.Parameter(torch.zeros(1, num_hidden))

        self.W_ph = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_hidden, num_classes)))
        self.b_p = nn.Parameter(torch.zeros(1, num_classes))

        self.h = []

    def forward(self, x):
        batch_size, seq_length = tuple(x.shape)
        self.h = []
        self.h.append(torch.zeros(batch_size, self.num_hidden, device=self.device, dtype=torch.float))

        for t in range(seq_length):
            x_t = x[:, t].reshape(-1, 1)
            h_t_1 = self.h[-1]
            c = x_t @ self.W_hx + h_t_1 @ self.W_hh + self.b_h
            h_t = torch.tanh(c)

            self.h.append(h_t)

        p = self.h[-1] @ self.W_ph + self.b_p

        return p

