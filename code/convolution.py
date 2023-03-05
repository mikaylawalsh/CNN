from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math


def conv2d(inputs, filters, strides, padding):
    """
    Performs 2D convolution given 4D inputs and filter Tensors.
    :param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
    :param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
    :param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
    :param padding: either "SAME" or "VALID", capitalization matters
    :return: outputs, Tensor with shape [num_examples, output_height, output_width, output_channels]
    """

    # how to i get these values
    num_examples = None
    in_height = None
    in_width = None
    input_in_channels = None

    filter_height = None
    filter_width = None
    filter_in_channels = None
    filter_out_channels = None

    num_examples_stride = None
    strideY = None
    strideX = None
    channels_stride = None

    # 1. assert inputs "in channels" is equal to filter's "in channels"
    assert(input_in_channels == filter_in_channels)
    # 2. check if padding is SAME or VALID
    # 3. if padding is SAME, calculate how much you need with (filter_size - 1)/2 and round using math.floor
    if padding == "SAME":
        pad_height = math.floor(filter_height - 1)/2
        pad_width = math.floor(filter_width - 1)/2
    else:
        pad_height = 0
        pad_width = 0
    # 4. use np.pad to pad input
    padded_input = np.pad(inputs, pad_width)
    # 5. create a NumPy array with the correct output dimensions (below)
    output_height = (in_height + 2*pad_height - filter_height) / strideY + 1
    output_width = (in_width + 2*pad_width - filter_width) / strideX + 1
    output = np.ones((output_height, output_width))
    # 6. update each element in the output as you perform the convolution operator for each image
    # 7. stop iterating when filter does not fit over rest of padding input
    # 8. perform the convolution per input channel and sum those dot products together
    while ...:
        for m in range(filter_height):  # i think i use my strides here to skip by multiple
            for n in range(filter_width):
                output[m][n] = inputs[m][n] * filters[m][n]
    # 9. return a tensor
    return tf.convert_to_tensor(output, dtype=tf.float32)
    # Cleaning padding input

    # Calculate output dimensions
    # height = (in_height + 2*padY - filter_height) / strideY + 1
    # width = (in_width + 2*padX - filter_width) / strideX + 1

    # PLEASE RETURN A TENSOR. HINT: tf.convert_to_tensor(your_array, dtype = tf.float32)


def same_test_0():
    '''
    Simple test using SAME padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[2, 2, 3, 3, 3], [0, 1, 3, 0, 3], [2, 3, 0, 1, 3], [
                    3, 3, 2, 1, 2], [3, 3, 0, 2, 3]], dtype=np.float32)
    imgs = np.reshape(imgs, (1, 5, 5, 1))
    filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
                                                     dtype=tf.float32,
                                                     stddev=1e-1),
                          name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="SAME")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="SAME")
    print("SAME_TEST_0:", "my conv2d:",
          my_conv[0][0][0], "tf conv2d:", tf_conv[0][0][0].numpy())


def valid_test_0():
    '''
    Simple test using VALID padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[2, 2, 3, 3, 3], [0, 1, 3, 0, 3], [2, 3, 0, 1, 3], [
                    3, 3, 2, 1, 2], [3, 3, 0, 2, 3]], dtype=np.float32)
    imgs = np.reshape(imgs, (1, 5, 5, 1))
    filters = tf.Variable(tf.random.truncated_normal([2, 2, 1, 1],
                                                     dtype=tf.float32,
                                                     stddev=1e-1),
                          name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
    print("VALID_TEST_0:", "my conv2d:",
          my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())


def valid_test_1():
    '''
    Simple test using VALID padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[3, 5, 3, 3], [5, 1, 4, 5], [2, 5, 0, 1],
                    [3, 3, 2, 1]], dtype=np.float32)
    imgs = np.reshape(imgs, (1, 4, 4, 1))
    filters = tf.Variable(tf.random.truncated_normal([3, 3, 1, 1],
                                                     dtype=tf.float32,
                                                     stddev=1e-1),
                          name="filters")
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
    print("VALID_TEST_1:", "my conv2d:",
          my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())


def valid_test_2():
    '''
    Simple test using VALID padding to check out differences between 
    own convolution function and TensorFlow's convolution function.

    NOTE: DO NOT EDIT
    '''
    imgs = np.array([[1, 3, 2, 1], [1, 3, 3, 1], [2, 1, 1, 3],
                    [3, 2, 3, 3]], dtype=np.float32)
    imgs = np.reshape(imgs, (1, 4, 4, 1))
    filters = np.array([[1, 2, 3], [0, 1, 0], [2, 1, 2]]).reshape(
        (3, 3, 1, 1)).astype(np.float32)
    my_conv = conv2d(imgs, filters, strides=[1, 1, 1, 1], padding="VALID")
    tf_conv = tf.nn.conv2d(imgs, filters, [1, 1, 1, 1], padding="VALID")
    print("VALID_TEST_1:", "my conv2d:",
          my_conv[0][0], "tf conv2d:", tf_conv[0][0].numpy())


def main():
    # TODO: Add in any tests you may want to use to view the differences between your and TensorFlow's output

    # 1. test to make sure results are very similar
    same_test_0()
    valid_test_0()
    valid_test_1()
    valid_test_2()
    return


if __name__ == '__main__':
    main()
