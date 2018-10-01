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

import csv
import os
import random
import time
from datetime import datetime
import argparse

import torch
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel


################################################################################

def save_checkpoint(state, model_path, rnd):
    torch.save(state, os.path.join(model_path, rnd, 'checkpoint.pth.tar'))


def train(config):
    # boilerplate
    rnd = datetime.now().strftime("%Y-%m-%d %H-%M") + '.' + config.txt_file.split('/')[-1].split('.txt')[0]
    os.makedirs(os.path.join(config.model_path, rnd), exist_ok=True)
    with open(os.path.join(config.model_path, rnd, 'summary.txt'), 'w+', encoding='utf-8') as summary_file, \
            open(os.path.join(config.model_path, rnd, 'stats.csv'), 'w+', encoding='utf-8') as stats_file, \
            open(os.path.join(config.model_path, rnd, 'hyperparams.txt'), 'w+', encoding='utf-8') as hyper_file:
        stats_writer = csv.DictWriter(stats_file, ['step', 'loss', 'acc'])
        stats_writer.writeheader()
        hyper_file.write(str(config))

        # Initialize the device which to run the model on
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # Initialize the dataset and data loader (note the +1)
        dataset = TextDataset(config.txt_file, config.seq_length, config.batch_size, config.train_steps)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=os.cpu_count())

        # Initialize the model that we are going to use
        model = TextGenerationModel(dataset.vocab_size, config.lstm_num_hidden, config.lstm_num_layers,
                                    1 - config.dropout_keep_prob, config.temperature).to(device=device)

        # Initialise weight with Xavier
        for name, param in model.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_normal_(param)

        # Setup the loss, optimizer, and scheduler
        criterion = torch.nn.NLLLoss().to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.learning_rate_step, config.learning_rate_decay)

        for step, (batch_inputs, batch_targets) in enumerate(data_loader):
            model.train()

            # schedule learning rate
            lr_scheduler.step()

            # Only for time measurement of step through network
            t1 = time.time()

            #######################################################
            # Add more code here ...
            #######################################################

            # zero out gradients
            model.zero_grad()

            # move to tensor one-hot encoding and initialise targets
            input_dim = (config.seq_length, config.batch_size, 1)
            one_hot_dim = (config.seq_length, config.batch_size, dataset.vocab_size)
            batch_inputs = torch.stack(batch_inputs, 0).to(device=device).view(*input_dim)
            batch_inputs = torch.zeros(*one_hot_dim, device=device, dtype=torch.float).scatter_(2, batch_inputs, 1.0)
            batch_targets = torch.stack(batch_targets, dim=1).to(device=device, dtype=torch.long)

            # forward pass
            batch_outputs, _, _ = model(batch_inputs)

            # negative log-likelihood loss with element-wise mean
            loss = criterion(batch_outputs, batch_targets)

            # backward pass
            loss.backward()

            # clip to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

            # update weights
            optimizer.step()

            # calculate accuracy and extract loss
            accuracy = float(torch.sum(torch.argmax(batch_outputs, dim=1) == batch_targets)) / (
                    config.seq_length * config.batch_size)
            loss = loss.item()

            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size / float(t2 - t1)

            if step % config.print_every == 0:
                now = datetime.now().strftime("%Y-%m-%d %H:%M")
                print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                      "Accuracy = {:.2f}, Loss = {:.3f}".format(now, step,
                                                                int(config.train_steps), config.batch_size,
                                                                examples_per_second, accuracy,
                                                                loss))

            # greedy sampling from the model
            if step % config.sample_every == 0:

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
                    'scheduler': lr_scheduler.state_dict()
                }, config.model_path, rnd)

                with torch.no_grad():
                    model.eval()

                    # get random seed
                    char_idx = random.choice(range(dataset.vocab_size))
                    ins = torch.zeros((1, 1, dataset.vocab_size), device=device, dtype=torch.float)
                    ins[0, 0, char_idx] = 1.0

                    # initialize hidden state and cell
                    h_t = torch.zeros(config.lstm_num_layers, 1, config.lstm_num_hidden, device=device,
                                      dtype=torch.float)
                    c_t = torch.zeros(config.lstm_num_layers, 1, config.lstm_num_hidden, device=device,
                                      dtype=torch.float)

                    # predict the next character and feed back into model
                    predictions = [char_idx]
                    for t in range(config.seq_length):
                        # compute energies
                        log_odds, (h_t, c_t), _ = model(ins, (h_t, c_t))
                        # obtain predicted sequence
                        pred_char = torch.argmax(log_odds, dim=1)
                        # append to generated sequence
                        predictions.append(pred_char.item())
                        # feed back into model
                        ins = torch.zeros((1, 1, dataset.vocab_size), device=device, dtype=torch.float)
                        ins[0, 0, pred_char.item()] = 1.0

                    sample = dataset.convert_to_string(predictions)
                    sample = "=============================={}\n{}\n==============================\n".format(step,
                                                                                                             sample)
                    summary_file.write(sample)

    print('Done training.')


################################################################################
################################################################################

if __name__ == "__main__":
    # Parse training configuration
    # parser = argparse.ArgumentParser()

    # Model params
    # parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    # parser.add_argument('--seq_length', type=int, default=30, help='Length of an input sequence')
    # parser.add_argument('--lstm_num_hidden', type=int, default=128, help='Number of hidden units in the LSTM')
    # parser.add_argument('--lstm_num_layers', type=int, default=2, help='Number of LSTM layers in the model')
    #
    # parser.add_argument('--temperature', type=float, default=1.0, help='Temperature parameter for energy annealing')
    #
    # # Training params
    # parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    # parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    #
    # # It is not necessary to implement the following three params, but it may help training.
    # parser.add_argument('--learning_rate_decay', type=float, default=0.96, help='Learning rate decay fraction')
    # parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')
    # parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help='Dropout keep probability')
    #
    # parser.add_argument('--train_steps', type=int, default=100000, help='Number of training steps')
    # parser.add_argument('--max_norm', type=float, default=5.0, help='--')
    #
    # # Misc params
    # parser.add_argument('--print_every', type=int, default=100, help='How often to print training progress')
    # parser.add_argument('--sample_every', type=int, default=100, help='How often to sample from the model')
    #
    # parser.add_argument('--model_path', type=str, default="./models/", help='Output path for models')

    # cfg = parser.parse_args()

    cfg = argparse.Namespace(batch_size=64, dropout_keep_prob=0.5, learning_rate=0.01, learning_rate_decay=0.9,
                             learning_rate_step=5000, lstm_num_hidden=128, lstm_num_layers=2, max_norm=5.0,
                             model_path='./models/', print_every=100, sample_every=100, seq_length=30, temperature=1.0,
                             train_steps=2000, txt_file='./data/us_constitution.txt')
    print(str(cfg))

    # Train the model
    train(cfg)
