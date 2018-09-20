"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from convnet_pytorch import ConvNet
from torch import nn, optim
import cifar10_utils
import numpy as np
import argparse
import torch
import csv
import os

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets, avg=True):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      targets: 1D int array of size [batch_size]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
      avg: Whether to average the results
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    acc = float(torch.sum(torch.argmax(predictions, dim=1) == targets))
    if avg:
        acc = acc / float(targets.shape[0])
    ########################
    # END OF YOUR CODE    #
    #######################

    return acc


def train():
    """
    Performs training and evaluation of ConvNet model.

    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    # DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ########################
    # PUT YOUR CODE HERE  #
    #######################

    os.makedirs('results', exist_ok=True)
    train_file = open('results/conv_pytorch_train.csv', 'w+')
    train_writer = csv.DictWriter(train_file, ['step', 'loss', 'acc'])
    train_writer.writeheader()
    eval_file = open('results/conv_pytorch_eval.csv', 'w+')
    eval_writer = csv.DictWriter(eval_file, ['step', 'loss', 'acc'])
    eval_writer.writeheader()

    # load data
    cifar_10 = cifar10_utils.get_cifar10(os.path.join('cifar10', 'cifar-10-batches-py'), one_hot=False)

    # initialise model and loss
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = ConvNet(n_channels=3, n_classes=10).to(device=device)
    criterion = nn.CrossEntropyLoss().to(device=device)
    optimizer = optim.Adam(model.parameters())

    # train model
    eval_step = 0
    for step in range(FLAGS.max_steps):
        model.train()
        # prepare data
        x, t = cifar_10['train'].next_batch(FLAGS.batch_size)
        x = torch.tensor(x, dtype=torch.float).to(device=device)
        t = torch.tensor(t, dtype=torch.long).to(device=device)

        # zero out gradients
        optimizer.zero_grad()

        # forward pass
        y = model.forward(x)
        loss_train = criterion(y, t)

        # backward pass
        loss_train.backward()

        # weight update
        optimizer.step()

        acc_train = accuracy(y, t)
        train_writer.writerow({
            'step': step,
            'loss': loss_train.data.item(),
            'acc': acc_train
        })
        # print('[Train Step: {:4}/{:4}] [Loss: {:5.4f}] [Accuracy: {:5.4f}]'.format(
        #     step + 1, FLAGS.max_steps, loss_train, acc_train
        # ))

        if step % FLAGS.eval_freq == 0:
            with torch.no_grad():
                model.eval()

                criterion_sum = nn.CrossEntropyLoss(reduction='sum').to(device=device)

                x_test, t_test = cifar_10['test'].images, cifar_10['test'].labels
                x_test = torch.tensor(x_test, dtype=torch.float).reshape(100, -1, 3, 32, 32).to(device=device)
                t_test = torch.tensor(t_test, dtype=torch.long).reshape(100, -1).to(device=device)
                loss_test = torch.zeros(1, dtype=torch.float).to(device=device)
                acc_test = torch.zeros(1, dtype=torch.float).to(device=device)
                for batch_idx in range(100):
                    y_batch_test = model.forward(x_test[batch_idx])
                    loss_test += criterion_sum.forward(y_batch_test, t_test[batch_idx])
                    acc_test += accuracy(y_batch_test, t_test[batch_idx], avg=False)
                avg_loss_test = float(loss_test) / (x_test.shape[0] * x_test.shape[1])
                avg_acc_test = float(acc_test) / (x_test.shape[0] * x_test.shape[1])

                print('=====')
                print('[Eval Step: {:2}/{:2}] [Loss: {:5.4f}] [Accuracy: {:5.4f}]'.format(
                    eval_step + 1, FLAGS.max_steps // FLAGS.eval_freq, avg_loss_test, avg_acc_test
                ))
                print('=====')
                eval_writer.writerow({
                    'step': eval_step,
                    'loss': avg_loss_test,
                    'acc': avg_acc_test
                })
                eval_step += 1
    train_file.close()
    eval_file.close()

    ########################
    # END OF YOUR CODE    #
    #######################


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()

    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)

    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()

    main()
