import cPickle as pkl
import os
import sys

import numpy as np
import prettytensor as pt
import tensorflow as tf
from progressbar import ETA, Bar, Percentage, ProgressBar

from infogan.models.regularized_gan import RegularizedGAN
from utils.python_utils import make_list

TINY = 1e-8


class LossUpdater(object):
    def __init__(self, datasets):
        self.datasets = make_list(datasets)

    def get_feed_dict(self):




class GANUpdater(LossUpdater):
    def __init__(self,
                 discrim_learning_rate=2e-4,
                 generator_learning_rate=1e-3,
                 **kwargs):
        d_optim = tf.train.AdamOptimizer(discrim_learning_rate, beta1=0.5)
        kwargs.setdefault('discrim_optimizer', d_optim)
        g_optim = tf.train.AdamOptimizer(generator_learning_rate, beta1=0.5)
        kwargs.setdefault('generator_optimizer', g_optim)
        super(LossUpdater, self).__init__(**kwargs)

    @staticmethod
    def get_discriminator_loss(real_d, fake_d):
        real = tf.log(real_d + TINY)
        fake = tf.log(1. - fake_d + TINY)
        return - tf.reduce_mean(real + fake)

    @staticmethod
    def get_generator_loss(fake_d):
        return - tf.reduce_mean(tf.log(fake_d + TINY))

    def update(self, sess, i, log_vars, all_log_vals):
        x, _ = self.dataset.train.next_batch(self.batch_size)
        feed_dict = {self.input_tensor: x}
        sess.run(self.generator_trainer, feed_dict)
        if i % self.gen_disc_update_ratio == 0:
            log_vals = sess.run([self.discriminator_trainer] + log_vars, feed_dict)[1:]
            all_log_vals.append(log_vals)
        return all_log_vals


class WassersteinUpdater(LossUpdater):
    def __init__(self,
                 discrim_learning_rate=5e-5,
                 generator_learning_rate=5e-5,
                 **kwargs):
        self.n_critic = 5
        self.discrim_weight_clip_by_value = [-0.01, 0.01]
        self.clip = None
        d_optim = tf.train.RMSPropOptimizer(discrim_learning_rate)
        kwargs.setdefault('discrim_optimizer', d_optim)
        g_optim = tf.train.RMSPropOptimizer(generator_learning_rate)
        kwargs.setdefault('generator_optimizer', g_optim)
        super(WassersteinGANTrainer, self).__init__(**kwargs)

    @staticmethod
    def get_discriminator_loss(real_d, fake_d):
        return tf.reduce_mean(real_d - fake_d)

    @staticmethod
    def get_generator_loss(fake_d):
        return tf.reduce_mean(fake_d)

    def update(self, sess, i, log_vars, all_log_vals):
        for _ in range(self.n_critic):
            x, _ = self.dataset.train.next_batch(self.batch_size)
            feed_dict = {self.input_tensor: x}
            log_vals = sess.run([self.discriminator_trainer] + log_vars,
                                feed_dict)[1:]
            if self.clip is None:
                self.clip = [tf.assign(
                    d, tf.clip_by_value(d, *self.discrim_weight_clip_by_value))
                             for d in self.d_vars]
            sess.run(self.clip)
            all_log_vals.append(log_vals)
        x, _ = self.dataset.train.next_batch(self.batch_size)
        feed_dict = {self.input_tensor: x}
        sess.run(self.generator_trainer, feed_dict)
        return all_log_vals
