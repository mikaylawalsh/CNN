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
    num_examples = inputs.shape[0]
    in_height = inputs.shape[1]
    in_width = inputs.shape[2]
    input_in_channels = inputs.shape[3]

    filter_height = filters.shape[0]
    filter_width = filters.shape[1]
    filter_in_channels = filters.shape[2]
    filter_out_channels = filters.shape[3]

    # should i make this [1, 1, 1, 1]
    num_examples_stride = strides[0]
    strideY = strides[1]
    strideX = strides[2]
    channels_stride = strides[3]

    # 1. assert inputs "in channels" is equal to filter's "in channels"
    assert(input_in_channels == filter_in_channels)
    # 2. check if padding is SAME or VALID
    # 3. if padding is SAME, calculate how much you need with (filter_size - 1)/2 and round using math.floor
    if padding == "SAME":
        if (in_height % strides[1] == 0):
            pad_along_height = max(filter_height - strideY, 0)
        else:
            pad_along_height = max(filter_height - (in_height % strideY), 0)
        if (in_width % strides[2] == 0):
            pad_along_width = max(filter_width - strideX, 0)
        else:
            pad_along_width = max(filter_width - (in_width % strideX), 0)

        pad_Ytop = pad_along_height // 2
        pad_Ybottom = pad_along_height - pad_Ytop
        pad_Xleft = pad_along_width // 2
        pad_Xright = pad_along_width - pad_Xleft
    else:
        pad_Xleft = 0
        pad_Xright = 0
        pad_Ytop = 0
        pad_Ybottom = 0
    
    # 4. use np.pad to pad input
    padded_input = np.pad(
        inputs, ((0, 0), (pad_Ytop, pad_Ybottom), (pad_Xleft, pad_Xright), (0, 0))) # is this right??
    in_height = padded_input.shape[1]
    in_width = padded_input.shape[2]
    # 5. create a NumPy array with the correct output dimensions (below)
    # should i be using padded dimensions 
    output_height = int(
        (in_height + 2*pad_Ytop - filter_height) / strideY + 1)
    output_width = int((in_width + 2*pad_Xleft - filter_width) / strideX + 1)
    output_dim1 = num_examples
    output_dim4 = filter_out_channels
    output = np.zeros((output_dim1, output_height, output_width, output_dim4)) 
    # 6. update each element in the output as you perform the convolution operator for each image
    # 7. stop iterating when filter does not fit over rest of padding input
    # 8. perform the convolution per input channel and sum those dot products together

    # You will want to iterate the entire height and width including padding, stopping when you cannot
    # fit a filter over the rest of the padding input. For convolution with many input channels, you will
    # want to perform the convolution per input channel and sum those dot products together.

    # i only want to alter inner two dimensions -- i need to sum something
    for i in range(0, num_examples, num_examples_stride):
        for j in range(filter_out_channels):
            for x in range(0, output_width, strideX):
                for y in range(0, output_height, strideY):
                    in_chunk = padded_input[i, x:x+filter_width, y:y+filter_height]
                    filt_chunk = filters[:, :, :, j]
                    output[i][x][y][j] = np.sum(np.multiply(in_chunk, filt_chunk))
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
    imgs = np.array([[2, 2, 3, 3, 3, 1], [0, 1, 3, 0, 3, 2], [2, 3, 0, 1, 3, 3], [
                    3, 3, 2, 1, 2, 3], [3, 3, 0, 2, 3, 2], [2, 3, 0, 1, 2, 3]], dtype=np.float32)
    imgs = np.reshape(imgs, (1, 6, 6, 1))
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
    print(filters)
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
    # valid_test_0()
    # valid_test_1()
    # valid_test_2()
    return


if __name__ == '__main__':
    main()
