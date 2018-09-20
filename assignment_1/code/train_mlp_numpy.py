"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv

import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 2e-3
MAX_STEPS_DEFAULT = 1500
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      targets: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    acc = np.sum(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1)) / targets.shape[0]
    ########################
    # END OF YOUR CODE    #
    #######################

    return acc


def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []

    ########################
    # PUT YOUR CODE HERE  #
    #######################
    os.makedirs('results', exist_ok=True)
    train_file = open('results/mlp_numpy_train.csv', 'w+')
    train_writer = csv.DictWriter(train_file, ['step', 'loss', 'acc'])
    train_writer.writeheader()
    eval_file = open('results/mlp_numpy_eval.csv', 'w+')
    eval_writer = csv.DictWriter(eval_file, ['step', 'loss', 'acc'])
    eval_writer.writeheader()

    # load data
    cifar_10 = cifar10_utils.get_cifar10(os.path.join('cifar10', 'cifar-10-batches-py'))

    # initialise model and loss
    model = MLP(n_inputs=3*32*32, n_hidden=dnn_hidden_units, n_classes=10)
    loss_fn = CrossEntropyModule()

    # train model
    eval_step = 0
    for step in range(FLAGS.max_steps):
        # forward pass
        x, t = cifar_10['train'].next_batch(FLAGS.batch_size)
        y = model.forward(x.reshape(FLAGS.batch_size, -1))

        # backward pass

        dloss = loss_fn.backward(y, t)
        model.backward(dloss)

        # stochastic gradient descent update
        for layer in model._layers:
            if type(layer).__name__ == 'LinearModule':
                layer.params['weight'] -= FLAGS.learning_rate * layer.grads['weight']
                layer.params['bias'] -= FLAGS.learning_rate * layer.grads['bias']

        loss_train = loss_fn.forward(y, t)
        acc_train = accuracy(y, t)
        train_writer.writerow({
            'step': step,
            'loss': loss_train,
            'acc': acc_train
        })

        print('[Train Step: {:4}/{:4}] [Loss: {:5.4f}] [Accuracy: {:5.4f}]'.format(
            step+1, FLAGS.max_steps, loss_train, acc_train
        ))

        if step % FLAGS.eval_freq == 0:

            x_test, t_test = cifar_10['test'].images, cifar_10['test'].labels
            y_test = model.forward(x_test.reshape(x_test.shape[0], -1))
            loss_test = loss_fn.forward(y_test, t_test)
            acc_test = accuracy(y_test, t_test)

            print('=====')
            print('[Eval Step: {:2}/{:2}] [Loss: {:5.4f}] [Accuracy: {:5.4f}]'.format(
                eval_step+1, FLAGS.max_steps // FLAGS.eval_freq, loss_test, acc_test
            ))
            print('=====')
            eval_writer.writerow({
                'step': eval_step,
                'loss': loss_test,
                'acc': acc_test
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
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
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
