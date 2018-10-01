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

class LSTM(nn.Module):

    def __init__(self, seq_length: int, input_dim: int, num_hidden: int, num_classes: int, batch_size: int,
                 device):
        super(LSTM, self).__init__()

        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.device = device

        self.W_gx = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(input_dim, num_hidden)))
        self.W_gh = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_hidden, num_hidden)))
        self.b_g = nn.Parameter(torch.zeros(1, num_hidden))

        self.W_ix = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(input_dim, num_hidden)))
        self.W_ih = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_hidden, num_hidden)))
        self.b_i = nn.Parameter(torch.zeros(1, num_hidden))

        self.W_fx = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(input_dim, num_hidden)))
        self.W_fh = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_hidden, num_hidden)))
        self.b_f = nn.Parameter(torch.zeros(1, num_hidden))

        self.W_ox = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(input_dim, num_hidden)))
        self.W_oh = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_hidden, num_hidden)))
        self.b_o = nn.Parameter(torch.zeros(1, num_hidden))

        self.W_hx = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(input_dim, num_hidden)))
        self.W_hh = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_hidden, num_hidden)))
        self.b_h = nn.Parameter(torch.zeros(1, num_hidden))

        self.W_ph = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(num_hidden, num_classes)))
        self.b_p = nn.Parameter(torch.zeros(1, num_classes))

        self.c = []
        self.h = []

    def forward(self, x):
        batch_size, seq_length = tuple(x.shape)
        self.c = []
        self.h = []
        self.c.append(torch.zeros(batch_size, self.num_hidden, device=self.device))
        self.h.append(torch.zeros(batch_size, self.num_hidden, device=self.device))

        for t in range(seq_length):
            x_t = x[:, t].reshape(-1, 1)
            c_t_1 = self.c[-1]
            h_t_1 = self.h[-1]
            g = torch.tanh(x_t @ self.W_gx + h_t_1 @ self.W_gh + self.b_g)
            i = torch.sigmoid(x_t @ self.W_ix + h_t_1 @ self.W_ih + self.b_i)
            f = torch.sigmoid(x_t @ self.W_fx + h_t_1 @ self.W_fh + self.b_f)
            o = torch.sigmoid(x_t @ self.W_ox + h_t_1 @ self.W_oh + self.b_o)
            c_t = g * i + c_t_1 * f
            h_t = torch.tanh(c_t) * o

            self.c.append(c_t)
            self.h.append(h_t)

        p = self.h[-1] @ self.W_ph + self.b_p

        return p
