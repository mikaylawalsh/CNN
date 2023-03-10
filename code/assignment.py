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

        self.batch_size = 64  # how to decide batch_size
        self.num_classes = 2 
        # Append losses to this list in training so you can visualize loss vs time in main
        self.loss_list = []

        # TODO: Initialize all hyperparameters

        # learning rate, kernel size, stride, batch size, epochs, padding, size of hidden layers
        # optimizer (adam), output dimensions
        self.learning_rate = 0.001
        self.epochs = 10
        self.stride = [1, 1, 1, 1]
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)
        self.padding = "SAME"

        # how many filters should i have? what size should they be?
        self.filters1 = tf.Variable(
            tf.random.truncated_normal(shape=[5, 5, 3, 16], stddev=.1))
        self.filters2 = tf.Variable(
            tf.random.truncated_normal([5, 5, 16, 20], stddev=.1))
        self.filters3 = tf.Variable(
            tf.random.truncated_normal([3, 3, 20, 20], stddev=.1))

        # TODO: Initialize all trainable parameters
        self.weights1 = tf.Variable(tf.random.truncated_normal(
            shape=[320, 128], stddev=.1))
        self.weights2 = tf.Variable(tf.random.truncated_normal(
            shape=[128, 64], stddev=.1))
        self.weights3 = tf.Variable(tf.random.truncated_normal(
            shape=[64, 2], stddev=.1))

        self.biases_conv1 = tf.Variable(
            tf.random.truncated_normal(shape=[16], stddev=.1))
        self.biases_conv2 = tf.Variable(
            tf.random.truncated_normal(shape=[20], stddev=.1))
        self.biases_conv3 = tf.Variable(
            tf.random.truncated_normal(shape=[20], stddev=.1))

        self.biases_dense1 = tf.Variable(
            tf.random.truncated_normal(shape=[128], stddev=.1))
        self.biases_dense2 = tf.Variable(
            tf.random.truncated_normal(shape=[64], stddev=.1))
        self.biases_dense3 = tf.Variable(
            tf.random.truncated_normal(shape=[2], stddev=.1))

        self.flatten = tf.keras.layers.Flatten()

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
        conv1 = tf.nn.conv2d(inputs, self.filters1, [1, 2, 2, 1], self.padding)

        bias_add1 = tf.nn.bias_add(conv1, self.biases_conv1)
        mean1, variance1 = tf.nn.moments(bias_add1, [0, 1, 2])
        batch_norm1 = tf.nn.batch_normalization(
            bias_add1, mean1, variance1, None, None, 1e-5)
        relu1 = tf.nn.relu(batch_norm1)
        max_pool1 = tf.nn.max_pool(relu1, [3, 3], [2, 2], self.padding)

        conv2 = tf.nn.conv2d(max_pool1, self.filters2, self.stride, self.padding)
        bias_add2 = tf.nn.bias_add(conv2, self.biases_conv2)
        mean2, variance2 = tf.nn.moments(bias_add2, [0, 1, 2])
        batch_norm2 = tf.nn.batch_normalization(
            bias_add2, mean2, variance2, None, None, 1e-5)
        relu2 = tf.nn.relu(batch_norm2)
        max_pool2 = tf.nn.max_pool(relu2, [2, 2], [2, 2], self.padding)
        if is_testing:
            conv3 = conv2d(max_pool2, self.filters3, self.stride, self.padding) 
        else:
            conv3 = tf.nn.conv2d(max_pool2, self.filters3, self.stride, self.padding)
        bias_add3 = tf.nn.bias_add(conv3, self.biases_conv3)
        mean3, variance3 = tf.nn.moments(bias_add3, [0, 1, 2])
        batch_norm3 = tf.nn.batch_normalization(
            bias_add3, mean3, variance3, None, None, 1e-5)
        relu3 = tf.nn.relu(batch_norm3)

        flat = self.flatten(relu3)

        dense1 = tf.matmul(flat, self.weights1) + self.biases_dense1
        relu4 = tf.nn.relu(dense1)
        drop1 = tf.nn.dropout(relu4, 0.3)

        dense2 = tf.matmul(drop1, self.weights2) + \
            self.biases_dense2
        relu5 = tf.nn.relu(dense2)
        drop2 = tf.nn.dropout(relu5, 0.3)

        logits = tf.matmul(drop2, self.weights3) + self.biases_dense3

        return logits

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.

        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """

        probs = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels, logits))

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
    #indices = [x for x in range(len(train_inputs))]
    indices = tf.random.shuffle(range(len(train_inputs)))
    shuffled_inputs = tf.gather(train_inputs, indices)
    shuffled_labels = tf.gather(train_labels, indices)

    # 2. make sure inputs are of correct size - (batch_size, width, height, in_channels)

    # 3. call tf.image.random_flip_left_right to increase accuracy -- call this on all?
    shuffled_inputs = tf.image.random_flip_left_right(shuffled_inputs)

    # 4. call call
    # 5. calculate loss within scope of tf.GradientTape
    # 6. use optimizer to apply gradients outside of gradient tape
    # 7. calculate accuracy
    # 8. print list of losses per batch

    # for b, b1 in enumerate(range(batch_size, x.shape[0] + 1, batch_size)):
    #     b0 = b1 - batch_size
    #     batch_metrics = self.batch_step(
    #         x[b0:b1], y[b0:b1], training=True)

    # b is batch number
    avg_acc = 0
    counter = 0
    for b1 in range(model.batch_size, shuffled_inputs.shape[0] + 1, model.batch_size):
    #for batch in range(0, len(shuffled_inputs), model.batch_size):
        b0 = b1 - model.batch_size
        batch_inputs = shuffled_inputs[b0:b1]
        batch_labels = shuffled_labels[b0:b1]

        with tf.GradientTape() as tape: 
            logits = model(batch_inputs, False)
            loss = model.loss(logits, batch_labels)
            model.loss_list.append(loss)
        grads = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        acc = model.accuracy(logits, batch_labels)
        avg_acc += acc
        counter += 1

        # print("loss:", loss)

    print("acc:", avg_acc/counter)  # ??

    return 


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
    avg_acc = 0
    for b1 in range(model.batch_size, test_inputs.shape[0] + 1, model.batch_size):
        b0 = b1 - model.batch_size
        batch_inputs = test_inputs[b0:b1]
        batch_labels = test_labels[b0:b1]

        logits = model(batch_inputs, True)
        loss = model.loss(logits, batch_labels)

        acc = model.accuracy(logits, batch_labels)
        avg_acc += acc

    return avg_acc/(test_inputs.shape[0]/model.batch_size)


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
    for _ in range(model.epochs):
        train(model, train_inputs, train_labels)

    # 4. call test method
    t = test(model, test_inputs, test_labels)
    print("TEST:", t)

    visualize_loss(model.loss_list)

    return


if __name__ == '__main__':
    main()
