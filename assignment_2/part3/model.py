# MIT License
#
# Copyright (c) 2017 Tom Runia
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

from typing import Tuple
import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, vocabulary_size: int, lstm_num_hidden: int, lstm_num_layers: int,
                 dropout: float, temperature: float):
        super(TextGenerationModel, self).__init__()

        # temperature used for annealing energies
        self.temperature = temperature

        # network architecture
        self.lstm = nn.LSTM(input_size=vocabulary_size, hidden_size=lstm_num_hidden,
                            num_layers=lstm_num_layers, dropout=dropout)
        self.map = nn.Linear(in_features=lstm_num_hidden, out_features=vocabulary_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x: torch.Tensor, hc_0: Tuple[torch.Tensor] = None):
        # feed input to lstm with or without hidden state
        if hc_0 is not None:
            h, hc_n = self.lstm(x, hc_0)
        else:
            h, hc_n = self.lstm(x)
        # transpose to batch-first and feed to linear map.
        p = self.map(h.transpose(0, 1))
        # anneal energies by temperature; a bit inefficient to divide everything even if T=1 but cleaner code
        p = p / self.temperature
        # transpose to number of classes as second dimension for compatibility with multi-dimensional NLLLoss
        log_odds = self.log_softmax(p).transpose(1, 2)

        # return the log-odds, the hidden state, and the energies
        return log_odds, hc_n, p
