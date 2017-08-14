from __future__ import absolute_import
from six.moves import range

import tensorflow as tf

from fcn_tf import tf_utils
from fcn_tf.networks import create_vgg16
from fcn_tf.networks import create_vgg19


def create_fcn(placeholder, keep_prob, classes, base='VGG19'):
    """
    Setup the main conv/deconv network
    """
    with tf.variable_scope('inference'):
        if base == 'VGG19':
            vgg_net = create_vgg19(placeholder)
            conv_final = vgg_net['relu5_4']
        elif base == 'VGG16':
            vgg_net = create_vgg16(placeholder)
            conv_final = vgg_net['relu5_3']
        else:
            raise NotImplementedError
        
        output = tf_utils.max_pool_2x2(conv_final)

        conv_shapes = [
            [7, 7, 512, 4096],
            [1, 1, 4096, 4096],
            [1, 1, 4096, classes]
        ]

        for i, conv_shape in enumerate(conv_shapes):
            name = 'conv%d' % (i + 6)
            with tf.variable_scope(name):
                W = tf_utils.weight_variable(conv_shape, name=name + '_w')
                b = tf_utils.bias_variable(conv_shape[-1:], name=name + '_b')
                output = tf_utils.conv2d(output, W, b)
            with tf.variable_scope('relu%d' % (i + 6)):
                if i < 2:
                    output = tf.nn.relu(output)
                    tf_utils.add_activation_summary(output, collections=['train'])
                    output = tf.nn.dropout(output, keep_prob=keep_prob)

        pool4 = vgg_net['pool4']
        pool3 = vgg_net['pool3']

        deconv_shapes = [
            tf.shape(pool4),
            tf.shape(pool3),
            tf.stack([
                tf.shape(placeholder)[0], tf.shape(placeholder)[1],
                tf.shape(placeholder)[2], classes
            ])
        ]

        W_shapes = [
            [4, 4, pool4.get_shape()[3].value, classes],
            [4, 4, pool3.get_shape()[3].value, pool4.get_shape()[3].value],
            [16, 16, classes, pool3.get_shape()[3].value]
        ]

        strides = [2, 2, 8]

        for i in range(3):
            name = 'deconv%d' % (i + 1)
            with tf.variable_scope(name):
                W = tf_utils.weight_variable(W_shapes[i], name=name + '_w')
                output = tf_utils.conv2d_transpose(
                    output, W, None,
                    output_shape=deconv_shapes[i], stride=strides[i])
            with tf.variable_scope('skip%d' % (i + 1)):
                if i < 2:
                    output = tf.add(output, vgg_net['pool%d' % (4 - i)])

        prediction = tf.argmax(output, dimension=3, name='prediction')
    
    return tf.expand_dims(prediction, dim=3), output

