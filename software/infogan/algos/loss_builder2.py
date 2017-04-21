import prettytensor as pt
import tensorflow as tf

from utils.python_utils import make_list

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


class AbstractLossBuilder(object):
    def __init__(self, models, g_optimizer=None, d_optimizer=None):
        self.models = make_list(models)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

        self.d_loss = None
        self.g_loss = None
        self.log_vars = []
        self.d_vars = None
        self.d_trainer = None
        self.g_trainer = None

    def get_g_feed_dict(self):
        raise NotImplementedError

    def get_d_feed_dict(self):
        raise NotImplementedError

    def init_opt(self):
        self.init_loss()
        self.init_optimizers()
        self.add_summaries()

    def add_summaries(self):
        self.log_vars.append(("discriminator_loss", self.d_loss))
        self.log_vars.append(("generator_loss", self.g_loss))
        for k, v in self.log_vars:
            tf.summary.scalar(name=k, tensor=v)
        self.add_train_samples_to_summary()

    def init_loss(self):
        raise NotImplementedError

    def add_train_samples_to_summary(self):
        raise NotImplementedError

    def init_optimizers(self):
        with pt.defaults_scope(phase=pt.Phase.train):
            all_vars = tf.trainable_variables()
            self.d_vars = [var for var in all_vars if var.name.startswith('d_')]
            g_vars = [var for var in all_vars if var.name.startswith('g_')]

            self.d_trainer = apply_optimizer(self.d_optimizer,
                                             losses=[self.d_loss],
                                             var_list=self.d_vars)

            self.g_trainer = apply_optimizer(self.g_optimizer,
                                             losses=[self.g_loss],
                                             var_list=g_vars)


class GANLossBuilder(AbstractLossBuilder):
    def __init__(self, model, loss, batch_size, **kwargs):
        """
        :type model: GANModel
        """
        self.loss = loss
        self.batch_size = batch_size

        self.fake_x = None
        super(GANLossBuilder, self).__init__(models=[model], **kwargs)

    @property
    def model(self):
        return self.models[0]

    def get_g_feed_dict(self):
        return self.model.get_g_feed_dict()

    def get_d_feed_dict(self):
        return self.model.get_d_feed_dict()

    def init_loss(self):
        with pt.defaults_scope(phase=pt.Phase.train):
            # (n, d)
            print('self.model.get_x_dist_flat()')
            x_dist_flat = self.model.get_x_dist_flat()
            print('self.model.get_x_dist_flat()2')
            x_dist_flat2 = self.model.get_x_dist_flat()
            print("x_dist_flat is x_dist_flat2: %s" % (x_dist_flat is x_dist_flat2))
            # print("\n".join(["Trainable variables"] + [v.name for v in tf.trainable_variables()]))
            # print("\n".join(["Global variables"] + [v.name for v in tf.global_variables()]))
            print('self.model.generate()')
            fake_x = self.model.generate()
            print(fake_x)
            print('self.model.generate()2')
            fake_x_2 = self.model.generate()
            print(fake_x_2)
            print(fake_x_2 is fake_x)
            # print("\n".join(["Trainable variables"] + [v.name for v in tf.trainable_variables()]))
            # print("\n".join(["Global variables"] + [v.name for v in tf.global_variables()]))
            print('self.model.get_x_dist_flat()')
            x_dist_flat = self.model.get_x_dist_flat()
            # print("\n".join(["Trainable variables"] + [v.name for v in tf.trainable_variables()]))
            # (n,)
            print('self.model.discriminate()')
            real_d = self.model.discriminate()
            # print("\n".join(["Trainable variables"] + [v.name for v in tf.trainable_variables()]))
            # print("\n".join(["Global variables"] + [v.name for v in tf.global_variables()]))
            print('self.model.discriminate(fake_x)')
            fake_d = self.model.discriminate(fake_x)

            self.log_vars.append(("max_real_d", tf.reduce_max(real_d)))
            self.log_vars.append(("min_real_d", tf.reduce_min(real_d)))
            self.log_vars.append(("max_fake_d", tf.reduce_max(fake_d)))
            self.log_vars.append(("min_fake_d", tf.reduce_min(fake_d)))
            self.fake_x = fake_x

            print('self.loss.get_d_loss(real_d, fake_d)')
            self.d_loss = self.loss.get_d_loss(real_d, fake_d)
            self.g_loss = self.loss.get_g_loss(fake_d)

        tf.add_to_collection("z_var", self.model.g_input)
        tf.add_to_collection("x_dist_flat", x_dist_flat)

    def add_train_samples_to_summary(self):
        with pt.defaults_scope(phase=pt.Phase.test):
            with tf.variable_scope("train_samples", reuse=True):
                self.model.get_train_samples()


class InfoGANLossBuilder(GANLossBuilder):
    def __init__(self, info_reg_coeff=1.0, **kwargs):
        self.info_reg_coeff = info_reg_coeff
        super(InfoGANLossBuilder, self).__init__(**kwargs)

    def init_loss(self):
        super(InfoGANLossBuilder, self).init_loss()
        with pt.defaults_scope(phase=pt.Phase.train):
            mi_est = tf.constant(0.)
            cross_ent = tf.constant(0.)
            reg_z = self.model.reg_z()
            fake_reg_z_dist_info = self.model.get_reg_dist_info(self.fake_x)
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
                self.d_loss -= self.info_reg_coeff * disc_mi_est
                self.g_loss -= self.info_reg_coeff * disc_mi_est

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
                self.d_loss -= self.info_reg_coeff * cont_mi_est
                self.g_loss -= self.info_reg_coeff * cont_mi_est

            for idx, dist_info in enumerate(self.model.reg_latent_dist.split_dist_info(fake_reg_z_dist_info)):
                if "stddev" in dist_info:
                    self.log_vars.append(("max_std_%d" % idx, tf.reduce_max(dist_info["stddev"])))
                    self.log_vars.append(("min_std_%d" % idx, tf.reduce_min(dist_info["stddev"])))

            self.log_vars.append(("MI", mi_est))
            self.log_vars.append(("CrossEnt", cross_ent))
