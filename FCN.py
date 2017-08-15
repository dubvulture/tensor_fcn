from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range

import os
import sys

import cv2
import dill
import numpy as np
import tensorflow as tf

from fcn_tf import tf_utils
from fcn_tf.networks import create_fcn


class FCN(object):

    def __init__(self,
                 classes,
                 logs_dir,
                 checkpoint=None):
        """
        :param classes: number of classes for classification
        :param logs_dir: directory for logs
        :param checkpoint: a CheckpointState from get_checkpoint_state
        """
        self.logs_dir = logs_dir
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.image = tf.placeholder(
            tf.float32, shape=[None, None, None, 3], name='image')
        self.annotation = tf.placeholder(
            tf.int32, shape=[None, None, None, 1], name='annotation')
        self.weight = tf.placeholder(
            tf.float32, shape=[None, None, None, 1], name='weight')

        self.lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        self.prediction, self.logits = create_fcn(self.image, self.keep_prob, classes)

        self.score = tf.nn.softmax(self.logits)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.loss_op = self._loss()
        self.train_op = self._training(global_step)

        self.checkpoint = checkpoint

    def _training(self, global_step):
        """
        Setup the training phase with Adam
        :param global_step: global step of training
        """
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads = optimizer.compute_gradients(self.loss_op)
        return optimizer.apply_gradients(grads, global_step=global_step)

    def _loss(self):
        """
        Setup the loss function
        """
        return tf.reduce_mean(
            tf.losses.sparse_softmax_cross_entropy(
                logits=self.logits,
                labels=self.annotation,
                weights=self.weight))

    def _setup_supervisor(self):
        """
        Setup the summary writer and variables
        """
        saver = tf.train.Saver(max_to_keep=20)
        sv = tf.train.Supervisor(
            logdir=self.logs_dir,
            save_summaries_secs=0,
            save_model_secs=0,
            saver=saver)

        # Restore checkpoint if given
        if self.checkpoint and self.checkpoint.model_checkpoint_path:
            print('Checkpoint found, loading model')
            with sv.managed_session() as sess:
                sv.saver.restore(sess, self.checkpoint.model_checkpoint_path)

        return sv

    def train(self,
              train_set,
              val_set=None,
              lr=1e-5,
              keep_prob=0.5,
              train_freq=10,
              val_freq=0,
              save_freq=500,
              max_steps=0):
        """
        :param train_set: instance of a DatasetReader subclass
        :param val_set: instance of a DatasetReader subclass
        :param lr: initial learning rate
        :param keep_prob: 1 - dropout
        :param train_freq: trace train_loss every train_freq iterations
        :param val_freq: trace val_loss every val_freq iterations
        :param save_freq: save model every save_freq iterations
        :param max_steps: max steps to perform
        """
        tf.summary.scalar('train_loss', self.loss_op, collections=['train'])
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var, collections=['train'])

        summ_train = tf.summary.merge_all(key='train')
        summ_val = tf.Summary()
        summ_val.value.add(tag='val_loss', simple_value=0)

        sv = self._setup_supervisor()

        with sv.managed_session() as sess:
            print('Starting training...')
            while not sv.should_stop():
                images, anns, weights, _ = train_set.next_batch()
                # Transform to match NN inputs
                images = images.astype(np.float32) / 255.
                anns = anns.astype(np.int32)
                feed = {
                    self.image: images,
                    self.annotation: anns,
                    self.weight: weights,
                    self.lr: lr,
                    self.keep_prob: keep_prob
                }
                sess.run(self.train_op, feed_dict=feed)

                step = sess.run(sv.global_step)

                if (step == max_steps) or ((step % train_freq) == 0):
                    loss, summary = sess.run(
                        [self.loss_op, summ_train],
                        feed_dict=feed)
                    sv.summary_computed(sess, summary, step)
                    print('Step %d\tTrain_loss: %g' % (step, loss))

                if ((val_set is not None) and (val_freq > 0) and
                        (((step % val_freq) == 0) or (step == max_steps))):
                    # Average loss on whole validation (sub)set
                    iters = val_set.size // val_set.batch_size
                    mean_loss = 0
                    for i in range(iters):
                        print('Running validation... %d/%d' % (i+1, iters), end='\r')
                        sys.stdout.flush()
                        images, anns, weights, _ = val_set.next_batch()
                        # Transform to match NN inputs
                        images = images.astype(np.float32) / 255.
                        anns = anns.astype(np.int32)
                        feed = {
                            self.image: images,
                            self.annotation: anns,
                            self.weight: weights,
                            self.keep_prob: 1.0
                        }
                        # no backpropagation
                        loss = sess.run(self.loss_op, feed_dict=feed)
                        mean_loss += loss

                    summ_val.value[0].simple_value = mean_loss / iters
                    sv.summary_computed(sess, summ_val, step)

                    print('\nStep %d\tValidation loss: %g' % (step, mean_loss / iters))

                if (step == max_steps) or ((save_freq > 0) and
                                           (step % save_freq) == 0):
                    # Save model
                    sv.saver.save(sess, self.logs_dir + 'model.ckpt', step)
                    print('Step %d\tModel saved.' % step)
                    # Save train & set dataset state
                    dill.dump(
                        train_set,
                        open(os.path.join(self.logs_dir, 'train_set.pkl'), 'wb'))
                    dill.dump(
                        val_set,
                        open(os.path.join(self.logs_dir, 'val_set.pkl'), 'wb'))

                if step == max_steps:
                    break

    def test(self, filenames, directory):
        """
        Run on images in directory without their groundtruth
        :param filenames:
        :param directory:
        """
        sv = self._setup_supervisor()
        
        with sv.managed_session() as sess:
            for i, fname in enumerate(filenames):
                in_image = cv2.imread(os.path.join(directory, fname))
                # pad image to the nearest multiple of 32
                dy, dx = tf_utils.get_pad(in_image, mul=32)
                in_image = tf_utils.pad(in_image, dy, dx)
                # batch size = 1
                in_image = np.expand_dims(in_image, axis=0)
                in_image = in_image.astype(np.float32) / 255.

                feed = {self.image: in_image, self.keep_prob: 1.0}
                pred, score = sess.run([self.prediction, self.score], feed_dict=feed)
                print('Evaluated image\t' + fname)

                # squeeze dims and undo padding
                dy = pred.shape[1] - dy
                dx = pred.shape[2] - dx
                output = np.squeeze(pred, axis=(0,3))[:dy, :dx]
                out_dir = os.path.join(self.logs_dir, 'output/')
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

                tf_utils.save_image(
                    np.uint8(output * 255),
                    out_dir,
                    name=fname)

