import cPickle as pkl
import os
import sys

import numpy as np
import prettytensor as pt
import tensorflow as tf
from progressbar import ETA, Bar, Percentage, ProgressBar

from infogan.models.regularized_gan import RegularizedGAN

TINY = 1e-8


def apply_optimizer(optimizer, losses, var_list, clip_by_value=None,
                    name='cost'):
    total_loss = tf.add_n(losses, name=name)
    grads_and_vars = optimizer.compute_gradients(total_loss, var_list=var_list)
    if clip_by_value is not None:
        clipped_grads_and_vars = []
        for g, v in grads_and_vars:
            cg = None
            if g is not None:
                cg = tf.clip_by_value(g, *clip_by_value)
            clipped_grads_and_vars.append((cg, v))
        grads_and_vars = clipped_grads_and_vars
    train_op = optimizer.apply_gradients(grads_and_vars)
    return train_op


class LossBuilder(object):
    def __init__(self,
                 model,
                 loss,
                 dataset,
                 batch_size,
                 discrim_optimizer,
                 generator_optimizer,
                 generator_grad_clip_by_value=None,
                 discrim_grad_clip_by_value=None,
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.loss = loss
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator_optimizer = generator_optimizer
        self.discrim_optimizer = discrim_optimizer
        self.discrim_grad_clip_by_value = discrim_grad_clip_by_value
        self.generator_grad_clip_by_value = generator_grad_clip_by_value

        self.discriminator_trainer = None
        self.generator_trainer = None
        self.g_input = None
        self.d_input = None
        self.log_vars = []
        self.discrim_loss = None
        self.generator_loss = None
        self.fake_reg_z_dist_info = None

    def get_feed_dict(self):
        raise NotImplementedError

    def prepare_g_input(self):
        raise NotImplementedError

    def prepare_d_input(self):
        d_input_shape = [self.batch_size, self.dataset.image_dim]
        # d_input_shape = [self.batch_size, self.d_input_dim]
        self.d_input = tf.placeholder(tf.float32, d_input_shape)

    def init_opt(self):
        self.init_loss()
        self.add_train_samples_to_summary()
        self.init_optimizers()
        for k, v in self.log_vars:
            tf.summary.scalar(name=k, tensor=v)

    def init_loss(self):
        self.prepare_g_input()
        self.prepare_d_input()

        with pt.defaults_scope(phase=pt.Phase.train):
            # (n, d)
            fake_x, _, x_dist_flat = self.model.generate(self.g_input)
            # (n,)
            real_d, _, _, _ = self.model.discriminate(self.d_input)
            fake_d, _, fake_reg_z_dist_info, _ = self.model.discriminate(fake_x)

            self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
            self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
            self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
            self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))
            self.fake_reg_z_dist_info = fake_reg_z_dist_info

            self.discrim_loss = self.loss.get_d_loss(real_d, fake_d)
            self.generator_loss = self.loss.get_g_loss(fake_d)

            self.log_vars.append(("discriminator_loss", self.discrim_loss))
            self.log_vars.append(("generator_loss", self.generator_loss))

        tf.add_to_collection("z_var", self.g_input)
        tf.add_to_collection("x_dist_flat", x_dist_flat)

    def add_train_samples_to_summary(self):
        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("train_samples", reuse=True):
                self.model.get_train_samples()

    def init_optimizers(self):
        with pt.defaults_scope(phase=pt.Phase.train):
            all_vars = tf.trainable_variables()
            self.d_vars = [var for var in all_vars if var.name.startswith('d_')]
            self.g_vars = [var for var in all_vars if var.name.startswith('g_')]


            self.discriminator_trainer = apply_optimizer(self.discrim_optimizer,
                                                         losses=[self.discrim_loss],
                                                         var_list=self.d_vars,
                                                         clip_by_value=self.discrim_grad_clip_by_value)

            self.generator_trainer = apply_optimizer(self.generator_optimizer,
                                                     losses=[self.generator_loss],
                                                     var_list=self.g_vars,
                                                     clip_by_value=self.generator_grad_clip_by_value)


class GANLossBuilder(LossBuilder):
    def get_feed_dict(self):
        x, _ = self.dataset.train.next_batch(self.batch_size)
        return {self.d_input: x}

    def prepare_g_input(self):
        self.g_input = self.model.latent_dist.sample_prior(self.batch_size)


class InfoGANLossBuilder(GANLossBuilder):
    def __init__(self, info_reg_coeff=1.0, **kwargs):
        self.info_reg_coeff = info_reg_coeff
        super(InfoGANLossBuilder, self).__init__(**kwargs)

    def init_loss(self):
        super(InfoGANLossBuilder, self).init_loss()
        with pt.defaults_scope(phase=pt.Phase.train):
            mi_est = tf.constant(0.)
            cross_ent = tf.constant(0.)
            reg_z = self.model.reg_z(self.g_input)
            # compute for discrete and continuous codes separately
            # discrete:
            if len(self.model.reg_disc_latent_dist.dists) > 0:
                disc_reg_z = self.model.disc_reg_z(reg_z)
                disc_reg_dist_info = self.model.disc_reg_dist_info(self.fake_reg_z_dist_info)
                disc_log_q_c_given_x = self.model.reg_disc_latent_dist.logli(disc_reg_z, disc_reg_dist_info)
                disc_log_q_c = self.model.reg_disc_latent_dist.logli_prior(disc_reg_z)
                disc_cross_ent = tf.reduce_mean(-disc_log_q_c_given_x)
                disc_ent = tf.reduce_mean(-disc_log_q_c)
                disc_mi_est = disc_ent - disc_cross_ent
                mi_est += disc_mi_est
                cross_ent += disc_cross_ent
                self.log_vars.append(("MI_disc", disc_mi_est))
                self.log_vars.append(("CrossEnt_disc", disc_cross_ent))
                self.discrim_loss -= self.info_reg_coeff * disc_mi_est
                self.generator_loss -= self.info_reg_coeff * disc_mi_est

            if len(self.model.reg_cont_latent_dist.dists) > 0:
                cont_reg_z = self.model.cont_reg_z(reg_z)
                cont_reg_dist_info = self.model.cont_reg_dist_info(self.fake_reg_z_dist_info)
                cont_log_q_c_given_x = self.model.reg_cont_latent_dist.logli(cont_reg_z, cont_reg_dist_info)
                cont_log_q_c = self.model.reg_cont_latent_dist.logli_prior(cont_reg_z)
                cont_cross_ent = tf.reduce_mean(-cont_log_q_c_given_x)
                cont_ent = tf.reduce_mean(-cont_log_q_c)
                cont_mi_est = cont_ent - cont_cross_ent
                mi_est += cont_mi_est
                cross_ent += cont_cross_ent
                self.log_vars.append(("MI_cont", cont_mi_est))
                self.log_vars.append(("CrossEnt_cont", cont_cross_ent))
                self.discrim_loss -= self.info_reg_coeff * cont_mi_est
                self.generator_loss -= self.info_reg_coeff * cont_mi_est

            for idx, dist_info in enumerate(self.model.reg_latent_dist.split_dist_info(self.fake_reg_z_dist_info)):
                if "stddev" in dist_info:
                    self.log_vars.append(("max_std_%d" % idx, tf.reduce_max(dist_info["stddev"])))
                    self.log_vars.append(("min_std_%d" % idx, tf.reduce_min(dist_info["stddev"])))

            self.log_vars.append(("MI", mi_est))
            self.log_vars.append(("CrossEnt", cross_ent))
