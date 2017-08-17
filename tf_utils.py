from __future__ import absolute_import
from __future__ import division

import os

import cv2
import numpy as np
import tensorflow as tf



def save_image(image, save_dir, name):
    """ 
    :param image: numpy array
    :param save_dir:
    :param name:
    """
    if issubclass(image.dtype.type, np.floating):
        image = np.uint8(image * 255)
    cv2.imwrite(os.path.join(save_dir, name + '.png'), image)


def get_pad(image, mul=32):
    """
    Return padding values needed to reach nearest multiple of mul
    """
    shape = np.array(image.shape[:2])
    return np.int32(np.ceil(shape / mul) * mul - shape)


def pad(image, dy, dx, val=0):
    """
    Pad image bottom and right with dy and dx
    """
    return np.pad(image,
                  ((0, dy), (0, dx), (0, 0)),
                  'constant',
                  constant_values=val)


def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init,  shape=weights.shape)
    return var


def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)


def conv2d(x, W, b, stride=1, pad=True):
    padding = 'SAME' if pad else 'VALID'
    conv = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.bias_add(conv, b)


def conv2d_transpose(x, W, b, output_shape=None, stride=2):
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]
    conv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')
    return conv


def leaky_relu(x, alpha=0.0, name=""):
    return tf.maximum(alpha * x, x, name)


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def add_to_regularization_and_summary(var):
    if var is not None:
        tf.summary.histogram(var.op.name, var)
        tf.add_to_collection("reg_loss", tf.nn.l2_loss(var))


def add_activation_summary(var, collections=None):
    if var is not None:
        tf.summary.histogram(var.op.name + "/activation", var, collections=collections)
        tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var), collections=collections)


def add_gradient_summary(grad, var, collections=None):
    if grad is not None:
        tf.summary.histogram(var.op.name + "/gradient", grad, collections=collections)

