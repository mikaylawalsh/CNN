from __future__ import absolute_import
from re import L
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random
import math

# ensures that we run only on cpu
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = None
        self.num_classes = None
        # Append losses to this list in training so you can visualize loss vs time in main
        self.loss_list = []

        # TODO: Initialize all hyperparameters

        # learning rate, kernel size, stride, batch size, epochs, padding, size of hidden layers
        # optimizer (adam), output dimensions
        self.learning_rate = 0.01
        self.epochs = 20
        self.stride = 1
        self.optimizer = tf.keras.optimizers.Adam()
        self.padding = "SAME"  # valid or same but thats not an option

        # TODO: Initialize all trainable parameters
        self.weights = tf.Variable(tf.random.truncated_normal(
            [100, self.num_classes], stddev=.1))  # shape right?
        self.biases = tf.Variable(
            tf.random.truncated_normal([self.num_classes], stddev=.1))
        # how many filters should i have? what size should they be?
        self.filters = tf.Variable(
            tf.random.truncated_normal([5, 5, 3, 16], stddev=.1))

        self.flatten = tf.keras.layers.Flatten()

        # weights, biases, kernel values

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.

        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        # add more layers later..basic model

        if is_testing:
            lay1 = conv2d(inputs, self.filters, self.stride, self.padding)
        else:
            # should stride be [1, 1, 1, 1] or 2? conflicting
            lay1 = tf.nn.conv2d(inputs, self.filters, 2, "SAME")

        # what goes here for value? correct addition?
        lay1 += tf.nn.bias_add(lay1, self.biases)
        mean, variance = tf.nn.moments(inputs, [0, 1, 2])  # ??
        lay1 = tf.nn.batch_normalization(
            lay1, mean, variance, None, None, 1e-5)  # what goes as input
        lay2 = tf.nn.relu(lay1)  # ?
        lay3 = tf.nn.max_pool(lay2, 3, 2, "SAME")
        logits = tf.matmul(lay3, self.weights) + self.biases

        return logits
        # is output shape correct?

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.

        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        probs = tf.nn.softmax_cross_entropy_with_logits(labels, logits)

        return probs

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.

        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT

        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(
            tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.

    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''

    # 1. shuffle inputs (tf.random.shuffle and tf.gather) -- i don't understand how zip would work here...

    # indices = []
    # for x in range(len(train_inputs)):
    #     indices.append(x)
    indices = [x for x in range(len(train_inputs))]
    indices = tf.random.shuffle(indices)
    shuffled_inputs = tf.gather(train_inputs, indices)
    shuffled_labels = tf.gather(train_labels, indices)

    # 2. make sure inputs are of correct size - (batch_size, width, height, in_channels)
    # assert(shuffled_inputs.shape == (self.batch_size, ))

    # 3. call tf.image.random_flip_left_right to increase accuracy -- call this on all?
    # shuffled_inputs = tf.image.random_flip_left_right(shuffled_inputs)

    # 4. call call
    # 5. calculate loss within scope of tf.GradientTape
    with tf.GradientTape as tape:
        logits = model.call(shuffled_inputs, False)
        loss = model.loss(logits, shuffled_labels)
        model.loss_list.append(loss)

    # 6. use optimizer to apply gradients outside of gradient tape
    grads = tape.gradient(loss, model.weights)
    model.optimizer.apply_gradients(model.weights, grads)
    # 7. calculate accuracy
    acc = model.accuracy(logits, shuffled_labels)

    # 8. print list of losses per batch
    return {'loss': loss, 'acc': acc}


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.

    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """
    logits = model.call(test_inputs, False)
    loss = model.loss(logits, test_labels)

    acc = model.accuracy(logits, test_labels)

    return {'loss': loss, 'acc': acc}


def visualize_loss(losses):
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
    """
    Uses Matplotlib to visualize the correct and incorrect results of our model.
    :param image_inputs: image data from get_data(), limited to 50 images, shape (50, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (50, num_classes)
    :param image_labels: the labels from get_data(), shape (50, num_classes)
    :param first_label: the name of the first class, "cat"
    :param second_label: the name of the second class, "dog"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, two plots should pop-up, one for correct results,
    one for incorrect results
    """
    # Helper function to plot images into 10 columns
    def plotter(image_indices, label):
        nc = 10
        nr = math.ceil(len(image_indices) / 10)
        fig = plt.figure()
        fig.suptitle(
            "{} Examples\nPL = Predicted Label\nAL = Actual Label".format(label))
        for i in range(len(image_indices)):
            ind = image_indices[i]
            ax = fig.add_subplot(nr, nc, i+1)
            ax.imshow(image_inputs[ind], cmap="Greys")
            pl = first_label if predicted_labels[ind] == 0.0 else second_label
            al = first_label if np.argmax(
                image_labels[ind], axis=0) == 0 else second_label
            ax.set(title="PL: {}\nAL: {}".format(pl, al))
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.tick_params(axis='both', which='both', length=0)

    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    # Separate correct and incorrect images
    correct = []
    incorrect = []
    for i in range(num_images):
        if predicted_labels[i] == np.argmax(image_labels[i], axis=0):
            correct.append(i)
        else:
            incorrect.append(i)

    plotter(correct, 'Correct')
    plotter(incorrect, 'Incorrect')
    plt.show()


def main():
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and 
    test your model for a number of epochs. We recommend that you train for
    10 epochs and at most 25 epochs. 

    CS1470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=70%.

    CS2470 students should receive a final accuracy 
    on the testing examples for cat and dog of >=75%.

    :return: None
    '''

    # 1. get train and test data
    train_inputs, train_labels = get_data(
        "/Users/mikaylawalsh/Desktop/deep_learning/hw3-mikaylawalsh/data/train", 3, 5)
    test_inputs, test_labels = get_data(
        "/Users/mikaylawalsh/Desktop/deep_learning/hw3-mikaylawalsh/data/test", 3, 5)

    # 2. initialize model
    model = Model()
    # 3. train it for many epochs (10) - use for loop
    for x in range(model.epochs):
        train(model, train_inputs, train_labels)
    # 4. call test method
    test(model, test_inputs, test_labels)

    return


if __name__ == '__main__':
    main()
