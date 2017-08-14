from __future__ import absolute_import

import os

import numpy as np
import scipy.io as sio
import tensorflow as tf

from fcn_tf import tf_utils


MODEL_16 = 'http://www.vlfeat.org/matconvnet/models/beta16/' \
            'imagenet-vgg-verydeep-16.mat'

MODEL_19 = 'http://www.vlfeat.org/matconvnet/models/beta16/' \
            'imagenet-vgg-verydeep-19.mat'

LAYERS_16 = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3'
)

LAYERS_19 = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
    'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
    'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
    'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4'
)


def _get_model_attr(model_dir, model_type):
    """
    Get the model file if needed and return VGG19 weights and mean pixel
    """
    maybe_download_and_extract(model_dir, model_type)
    filename = model_type.split("/")[-1]
    filepath = os.path.join(model_dir, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found.")
    
    data = sio.loadmat(filepath)
    mean = data['normalization']['averageImage'][0,0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = np.squeeze(data['layers'])

    return weights, mean_pixel


def _setup_net(placeholder, layers, weights, mean_pixel):
    """
    Returns the cnn built with given weights and normalized with mean_pixel
    """
    net = {}
    placeholder -= mean_pixel
    for i, name in enumerate(layers):
        kind = name[:4]
        with tf.variable_scope(name):
            if kind == 'conv':
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: [width, height, in_channels, out_channels]
                # tensorflow: [height, width, in_channels, out_channels]
                kernels = tf_utils.get_variable(
                    np.transpose(kernels, (1, 0, 2, 3)),
                    name=name + "_w")
                bias = tf_utils.get_variable(
                    bias.reshape(-1),
                    name=name + "_b")
                placeholder = tf_utils.conv2d(placeholder, kernels, bias)
            elif kind == 'relu':
                placeholder = tf.nn.relu(placeholder, name=name)
                tf_utils.add_activation_summary(placeholder, collections=['train'])
            elif kind == 'pool':
                placeholder = tf_utils.max_pool_2x2(placeholder)
            net[name] = placeholder

    return net


def create_vgg16(placeholder, model_dir='fcn_tf/networks/Model_zoo/'):
    """
    :param placeholder: tf.placeholder where we will operate
    :param model_dir: directory where to find the model .mat file
    """
    global LAYERS_16
    global MODEL_16
    return _setup_net(placeholder, LAYERS_16, *_get_model_attr(model_dir, MODEL_16))


def create_vgg19(placeholder, model_dir='fcn_tf/networks/Model_zoo/'):
    """
    :param placeholder: tf.placeholder where we will operate
    :param model_dir: directory where to find the model .mat file
    """
    global LAYERS_19
    global MODEL_19
    return _setup_net(placeholder, LAYERS_19, *_get_model_attr(model_dir, MODEL_19))


def maybe_download_and_extract(dir_path, url_name, is_tarfile=False, is_zipfile=False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        statinfo = os.stat(filepath)
        print('\nSuccesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)

