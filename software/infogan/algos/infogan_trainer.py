from infogan.models.regularized_gan import RegularizedGAN
import prettytensor as pt
import tensorflow as tf
import numpy as np
from progressbar import ETA, Bar, Percentage, ProgressBar
from infogan.misc.distributions import Bernoulli, Gaussian, Categorical
import sys

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


class GANTrainer(object):
    def __init__(self,
                 model,
                 batch_size,
                 discrim_optimizer,
                 generator_optimizer,
                 dataset=None,
                 exp_name="experiment",
                 log_dir="logs",
                 checkpoint_dir="ckt",
                 max_epoch=100,
                 updates_per_epoch=100,
                 snapshot_interval=500,
                 info_reg_coeff=1.0,
                 gen_disc_update_ratio=1,
                 generator_grad_clip_by_value=None,
                 discrim_grad_clip_by_value=None,
                 ):
        """
        :type model: RegularizedGAN
        """
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator_optimizer = generator_optimizer
        self.discrim_optimizer = discrim_optimizer
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.snapshot_interval = snapshot_interval
        self.updates_per_epoch = updates_per_epoch
        self.info_reg_coeff = info_reg_coeff
        self.discriminator_trainer = None
        self.generator_trainer = None
        self.input_tensor = None
        self.log_vars = []
        self.gen_disc_update_ratio = gen_disc_update_ratio
        self.discrim_grad_clip_by_value = discrim_grad_clip_by_value
        self.generator_grad_clip_by_value = generator_grad_clip_by_value

    def get_discriminator_loss(self, real_d, fake_d):
        raise NotImplementedError

    def get_generator_loss(self, fake_d):
        raise NotImplementedError

    def init_opt(self):
        self.input_tensor = input_tensor = tf.placeholder(tf.float32, [self.batch_size, self.dataset.image_dim])

        with pt.defaults_scope(phase=pt.Phase.train):
            # (n, input_dim)
            z_var = self.model.latent_dist.sample_prior(self.batch_size)
            fake_x, _ = self.model.generate(z_var)  # (n, d)
            real_d, _, _, _ = self.model.discriminate(input_tensor)  # (n,)
            fake_d, _, fake_reg_z_dist_info, _ = self.model.discriminate(fake_x)

            reg_z = self.model.reg_z(z_var)

            discrim_loss = self.get_discriminator_loss(real_d, fake_d)
            generator_loss = self.get_generator_loss(fake_d)

            self.log_vars.append(("discriminator_loss", discrim_loss))
            self.log_vars.append(("generator_loss", generator_loss))

            mi_est = tf.constant(0.)
            cross_ent = tf.constant(0.)

            # compute for discrete and continuous codes separately
            # discrete:
            if len(self.model.reg_disc_latent_dist.dists) > 0:
                disc_reg_z = self.model.disc_reg_z(reg_z)
                disc_reg_dist_info = self.model.disc_reg_dist_info(fake_reg_z_dist_info)
                disc_log_q_c_given_x = self.model.reg_disc_latent_dist.logli(disc_reg_z, disc_reg_dist_info)
                disc_log_q_c = self.model.reg_disc_latent_dist.logli_prior(disc_reg_z)
                disc_cross_ent = tf.reduce_mean(-disc_log_q_c_given_x)
                disc_ent = tf.reduce_mean(-disc_log_q_c)
                disc_mi_est = disc_ent - disc_cross_ent
                mi_est += disc_mi_est
                cross_ent += disc_cross_ent
                self.log_vars.append(("MI_disc", disc_mi_est))
                self.log_vars.append(("CrossEnt_disc", disc_cross_ent))
                discrim_loss -= self.info_reg_coeff * disc_mi_est
                generator_loss -= self.info_reg_coeff * disc_mi_est

            if len(self.model.reg_cont_latent_dist.dists) > 0:
                cont_reg_z = self.model.cont_reg_z(reg_z)
                cont_reg_dist_info = self.model.cont_reg_dist_info(fake_reg_z_dist_info)
                cont_log_q_c_given_x = self.model.reg_cont_latent_dist.logli(cont_reg_z, cont_reg_dist_info)
                cont_log_q_c = self.model.reg_cont_latent_dist.logli_prior(cont_reg_z)
                cont_cross_ent = tf.reduce_mean(-cont_log_q_c_given_x)
                cont_ent = tf.reduce_mean(-cont_log_q_c)
                cont_mi_est = cont_ent - cont_cross_ent
                mi_est += cont_mi_est
                cross_ent += cont_cross_ent
                self.log_vars.append(("MI_cont", cont_mi_est))
                self.log_vars.append(("CrossEnt_cont", cont_cross_ent))
                discrim_loss -= self.info_reg_coeff * cont_mi_est
                generator_loss -= self.info_reg_coeff * cont_mi_est

            for idx, dist_info in enumerate(self.model.reg_latent_dist.split_dist_info(fake_reg_z_dist_info)):
                if "stddev" in dist_info:
                    self.log_vars.append(("max_std_%d" % idx, tf.reduce_max(dist_info["stddev"])))
                    self.log_vars.append(("min_std_%d" % idx, tf.reduce_min(dist_info["stddev"])))

            self.log_vars.append(("MI", mi_est))
            self.log_vars.append(("CrossEnt", cross_ent))

            all_vars = tf.trainable_variables()
            self.d_vars = [var for var in all_vars if var.name.startswith('d_')]
            self.g_vars = [var for var in all_vars if var.name.startswith('g_')]

            self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
            self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
            self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
            self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))

            self.discriminator_trainer = apply_optimizer(self.discrim_optimizer,
                                                         losses=[discrim_loss],
                                                         var_list=self.d_vars,
                                                         clip_by_value=self.discrim_grad_clip_by_value)

            self.generator_trainer = apply_optimizer(self.generator_optimizer,
                                                     losses=[generator_loss],
                                                     var_list=self.g_vars,
                                                     clip_by_value=self.generator_grad_clip_by_value)

            for k, v in self.log_vars:
                tf.summary.scalar(name=k, tensor=v)

        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("model", reuse=True) as scope:
                self.get_samples()

        tf.add_to_collection("z_var", z_var)
        tf.add_to_collection("generated", fake_x)

    def add_images_to_summary(self, z_var, images_name):
            _, x_dist_info = self.model.generate(z_var)

            # just take the mean image
            if isinstance(self.model.output_dist, Bernoulli):
                img_var = x_dist_info["p"]
            elif isinstance(self.model.output_dist, Gaussian):
                img_var = x_dist_info["mean"]
            else:
                raise NotImplementedError
            img_var = self.dataset.inverse_transform(img_var)
            rows = 10  # Number of rows and columns of images to generate
            # (n, h, w, c)
            img_var = tf.reshape(img_var, [self.batch_size] + list(self.dataset.image_shape))
            img_var = img_var[:rows * rows, :, :, :]
            # (rows, rows, h, w, c)
            imgs = tf.reshape(img_var, [rows, rows] + list(self.dataset.image_shape))
            stacked_img = []
            for row in xrange(rows):
                row_img = []
                for col in xrange(rows):
                    row_img.append(imgs[row, col, :, :, :])
                stacked_img.append(tf.concat(1, row_img))
            imgs = tf.concat(0, stacked_img)
            imgs = tf.expand_dims(imgs, 0)
            tf.summary.image(name=images_name, tensor=imgs)

    def get_samples(self):
        if len(self.model.reg_latent_dist.dists) > 0:
            return self.get_samples_with_reg_latent_dist()
        return self.get_samples_without_reg_latent_dist()

    def get_samples_without_reg_latent_dist(self):
        with tf.Session():
            # (n, d)
            z_var = self.model.nonreg_latent_dist.sample_prior(
                self.batch_size).eval()
        self.add_images_to_summary(z_var, 'image')

    def get_samples_with_reg_latent_dist(self):
        with tf.Session():
            # (n, d) with 10 * 10 samples + other samples
            fixed_noncat = np.concatenate([
                np.tile(
                    self.model.nonreg_latent_dist.sample_prior(10).eval(),
                    [10, 1]
                ),
                self.model.nonreg_latent_dist.sample_prior(self.batch_size - 100).eval(),
            ], axis=0)
            fixed_cat = np.concatenate([
                np.tile(
                    self.model.reg_latent_dist.sample_prior(10).eval(),
                    [10, 1]
                ),
                self.model.reg_latent_dist.sample_prior(self.batch_size - 100).eval(),
            ], axis=0)

        offset = 0
        for dist_idx, dist in enumerate(self.model.reg_latent_dist.dists):
            if isinstance(dist, Gaussian):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                c_vals = []
                for idx in xrange(10):
                    c_vals.extend([-1.0 + idx * 2.0 / 9] * 10)
                c_vals.extend([0.] * (self.batch_size - 100))
                vary_cat = np.asarray(c_vals, dtype=np.float32).reshape((-1, 1))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+1] = vary_cat
                offset += 1
            elif isinstance(dist, Categorical):
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(10):
                    cat_ids.extend([idx] * 10)
                cat_ids.extend([0] * (self.batch_size - 100))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+dist.dim] = lookup[cat_ids]
                offset += dist.dim
            elif isinstance(dist, Bernoulli):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(10):
                    cat_ids.extend([int(idx / 5)] * 10)
                cat_ids.extend([0] * (self.batch_size - 100))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+dist.dim] = np.expand_dims(np.array(cat_ids), axis=-1)
                # import ipdb; ipdb.set_trace()
                offset += dist.dim
            else:
                raise NotImplementedError
            z_var = tf.constant(np.concatenate([fixed_noncat, cur_cat], axis=1))
            # Images where each row is a different sample from dist
            # and each column a different non reg latent sample.
            name = 'image_{}_{}'.format(dist_idx, dist.__class__.__name__)
            self.add_images_to_summary(z_var, name)

    def update(self, sess, i, log_vars, all_log_vals):
        raise NotImplementedError

    def train(self):
        self.init_opt()

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            saver = tf.train.Saver()

            counter = 0

            log_vars = [x for _, x in self.log_vars]
            log_keys = [x for x, _ in self.log_vars]

            for epoch in range(self.max_epoch):
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                pbar.start()

                all_log_vals = []
                for i in range(self.updates_per_epoch):
                    pbar.update(i)
                    all_log_vals = self.update(sess, i, log_vars, all_log_vals)
                    counter += 1

                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                        print("Model saved in file: %s" % fn)

                # (n, h, w, c)
                x, _ = self.dataset.train.next_batch(self.batch_size)

                summary_str = sess.run(summary_op, {self.input_tensor: x})
                summary_writer.add_summary(summary_str, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                log_dict = dict(zip(log_keys, avg_log_vals))

                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                print("Epoch %d | " % (epoch) + log_line)
                sys.stdout.flush()
                if np.any(np.isnan(avg_log_vals)):
                    raise ValueError("NaN detected!")


class InfoGANTrainer(GANTrainer):
    def __init__(self,
                 discrim_learning_rate=2e-4,
                 generator_learning_rate=1e-3,
                 **kwargs):
        d_optim = tf.train.AdamOptimizer(discrim_learning_rate, beta1=0.5)
        kwargs.setdefault('discrim_optimizer', d_optim)
        g_optim = tf.train.AdamOptimizer(generator_learning_rate, beta1=0.5)
        kwargs.setdefault('generator_optimizer', g_optim)
        super(InfoGANTrainer, self).__init__(**kwargs)

    def get_discriminator_loss(self, real_d, fake_d):
        real = tf.log(real_d + TINY)
        fake = tf.log(1. - fake_d + TINY)
        return - tf.reduce_mean(real + fake)

    def get_generator_loss(self, fake_d):
        return - tf.reduce_mean(tf.log(fake_d + TINY))

    def update(self, sess, i, log_vars, all_log_vals):
        x, _ = self.dataset.train.next_batch(self.batch_size)
        feed_dict = {self.input_tensor: x}
        sess.run(self.generator_trainer, feed_dict)
        if i % self.gen_disc_update_ratio == 0:
            log_vals = sess.run([self.discriminator_trainer] + log_vars, feed_dict)[1:]
            all_log_vals.append(log_vals)
        return all_log_vals


class WassersteinGANTrainer(GANTrainer):
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

    def get_discriminator_loss(self, real_d, fake_d):
        return tf.reduce_mean(real_d - fake_d)

    def get_generator_loss(self, fake_d):
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
