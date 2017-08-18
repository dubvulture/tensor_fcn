from __future__ import absolute_import

import os

import tensorflow as tf

from tensor_fcn.fcn import TensorFCN
from tensor_fcn.dataset_reader import ADE_Dataset


"""
Just some simple examples showing how to use any of this.
"""


ade_dir = '/path/to/ADEChallengeData2016/'
logs_dir = os.path.abspath('/path/to/logs/')
ckpt = tf.train.get_checkpoint_state(logs_dir)
network = TensorFCN(151, logs_dir, checkpoint=ckpt)


def ade_train():
    # Batch Size = 4, Image/Crop Size = 256,  
    train_set = ADE_Dataset(os.path.join(ade_dir, 'train/'), 4, 256, crop=True)
    # Retrieve just a small subset 
    fnames = [i[:-4] for i in os.listdir(os.path.join(ade_dir, 'val/images/'))][:50]
    # Batch Size = 1, Image Size = 0 (variable, do not crop) 
    val_set = ADE_Dataset(os.path.join(ade_dir, 'val/'), 1, 0, crop=False, fnames=fnames)
    # Start training    
    network.train(train_set, val_set, lr=1e-5, train_freq=10, val_freq=100, save_freq=500, max_steps=500)


def test():
    test_dir = '/path/to/some/images/'
    files = [
        os.path.abspath(os.path.join(test_dir, f))
        for f in os.listdir(test_dir)
    ]
    network.test(files)

