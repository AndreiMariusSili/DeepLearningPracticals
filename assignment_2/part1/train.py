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

import argparse
import csv
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN
from lstm import LSTM


# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

################################################################################

def save_checkpoint(state, model_path, rnd):
    torch.save(state, os.path.join(model_path, rnd, 'checkpoint.pth.tar'))


def train(config):
    assert config.model_type in ('RNN', 'LSTM')

    # boilerplate
    rnd = config.model_type + '.' + str(config.input_length + 1)
    os.makedirs(os.path.join(config.model_path, rnd), exist_ok=True)
    with open(os.path.join(config.model_path, rnd, 'stats.csv'), 'w+', encoding='utf-8') as stats_file, \
            open(os.path.join(config.model_path, rnd, 'hyperparams.txt'), 'w+', encoding='utf-8') as hyper_file:
        stats_writer = csv.DictWriter(stats_file, ['step', 'loss', 'acc'])
        stats_writer.writeheader()
        hyper_file.write(str(config))

        # Initialize the device which to run the model on
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Initialize the model that we are going to use
        if config.model_type == 'RNN':
            model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes,
                               config.batch_size, device).to(device=device)
        elif config.model_type == 'LSTM':
            model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size,
                         device).to(device=device)
        else:
            raise ValueError('Unknown model type: {}'.format(config.model_type))

        # Initialize the dataset and data loader (note the +1)
        dataset = PalindromeDataset(config.input_length + 1, config.batch_size, config.train_steps)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=os.cpu_count())

        # Setup the loss and optimizer
        criterion = torch.nn.CrossEntropyLoss().to(device=device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):

            # Only for time measurement of step through network
            t1 = time.time()

            # zero out gradients
            model.zero_grad()

            # forward pass
            batch_inputs = batch_inputs.to(device=device)
            batch_targets = batch_targets.to(device=device)
            batch_outputs = model(batch_inputs)

            # compute loss
            loss = criterion(batch_outputs, batch_targets)

            # backward pass
            loss.backward()

            ############################################################################
            # QUESTION: what happens here and why?
            ############################################################################
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            ############################################################################

            # update weights
            optimizer.step()

            # calculate metrics
            accuracy = float(torch.sum(torch.argmax(batch_outputs, dim=1) == batch_targets).item()) / config.batch_size
            loss = loss.item()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size / float(t2 - t1)

            if step % 100 == 0:
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                                                                config.train_steps,
                                                                config.batch_size, examples_per_second,
                                                                accuracy, loss
                                                                ))

                # save statistics
                stats_writer.writerow({
                    'step': step,
                    'loss': loss,
                    'acc': accuracy
                })

    # save model
    save_checkpoint({
        'step': step + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, config.model_path, rnd)


################################################################################
################################################################################

if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--model_type', type=str, default="RNN", help="Model type, should be 'RNN' or 'LSTM'")
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    parser.add_argument('--model_path', type=str, default="./models", help="Output path for models")

    cfg = parser.parse_args()
    print(cfg)
    # Train the model
    train(cfg)
