import prettytensor as pt
import tensorflow as tf
import numpy as np

from infogan.misc.custom_ops import leaky_rectify
from infogan.misc.distributions import Product, Distribution, Gaussian, Categorical, Bernoulli
from utils.python_utils import make_list


class RegularizedGAN(object):
    def __init__(self, output_dist, latent_spec, batch_size, dataset,
                 final_activation=tf.nn.sigmoid):
        """
        Args:
            output_dist (Distribution): The output distiribution.
            latent_spec (list): List of latent distributions.
             [(Distribution, bool)]
             The boolean indicates if the distribution should be used for
             regularization.
            batch_size (int): The batch size
            dataset (DatasetIterator): used for image_shape and inverse_transform()
        """
        self.output_dist = output_dist
        self.latent_spec = latent_spec
        self.latent_dist = Product([x for x, _ in latent_spec])
        self.reg_latent_dist = Product([x for x, reg in latent_spec if reg])
        self.nonreg_latent_dist = Product([x for x, reg in latent_spec if not reg])
        self.batch_size = batch_size
        self.dataset = dataset
        self.image_shape = dataset.image_shape
        self.image_size = self.image_shape[0]
        assert all(isinstance(x, (Gaussian, Categorical, Bernoulli)) for x in self.reg_latent_dist.dists)

        self.reg_cont_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, Gaussian)])
        self.reg_disc_latent_dist = Product([x for x in self.reg_latent_dist.dists if isinstance(x, (Categorical, Bernoulli))])

        self.final_activation = final_activation
        self.build_network()

    def __getstate__(self):
        pickling_dict = self.__dict__.copy()
        del pickling_dict['discriminator_template']
        del pickling_dict['generator_template']
        del pickling_dict['encoder_template']
        del pickling_dict['final_activation']
        return pickling_dict

    def build_network(self):
        raise NotImplementedError

    def discriminate(self, x_var):
        d_out = self.discriminator_template.construct(input=x_var)
        if self.final_activation is not None:
            d = self.final_activation(d_out[:, 0])
        else:
            d = d_out[:, 0]
        reg_dist_flat = self.encoder_template.construct(input=x_var)
        reg_dist_info = self.reg_latent_dist.activate_dist(reg_dist_flat)
        return d, self.reg_latent_dist.sample(reg_dist_info), reg_dist_info, reg_dist_flat

    def generate(self, z_var):
        x_dist_flat = self.generator_template.construct(input=z_var)
        x_dist_info = self.output_dist.activate_dist(x_dist_flat)
        return self.output_dist.sample(x_dist_info), x_dist_info, x_dist_flat

    def disc_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(z_i)
        return self.reg_disc_latent_dist.join_vars(ret)

    def cont_reg_z(self, reg_z_var):
        ret = []
        for dist_i, z_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_var(reg_z_var)):
            if isinstance(dist_i, Gaussian):
                ret.append(z_i)
        return self.reg_cont_latent_dist.join_vars(ret)

    def disc_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, (Categorical, Bernoulli)):
                ret.append(dist_info_i)
        return self.reg_disc_latent_dist.join_dist_infos(ret)

    def cont_reg_dist_info(self, reg_dist_info):
        ret = []
        for dist_i, dist_info_i in zip(self.reg_latent_dist.dists, self.reg_latent_dist.split_dist_info(reg_dist_info)):
            if isinstance(dist_i, Gaussian):
                ret.append(dist_info_i)
        return self.reg_cont_latent_dist.join_dist_infos(ret)

    def reg_z(self, z_var):
        """ Return the variables with distribution bool == True (concatenated). """
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if reg_i:
                ret.append(z_i)
        return self.reg_latent_dist.join_vars(ret)

    def nonreg_z(self, z_var):
        ret = []
        for (_, reg_i), z_i in zip(self.latent_spec, self.latent_dist.split_var(z_var)):
            if not reg_i:
                ret.append(z_i)
        return self.nonreg_latent_dist.join_vars(ret)

    def reg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if reg_i:
                ret.append(dist_info_i)
        return self.reg_latent_dist.join_dist_infos(ret)

    def nonreg_dist_info(self, dist_info):
        ret = []
        for (_, reg_i), dist_info_i in zip(self.latent_spec, self.latent_dist.split_dist_info(dist_info)):
            if not reg_i:
                ret.append(dist_info_i)
        return self.nonreg_latent_dist.join_dist_infos(ret)

    def combine_reg_nonreg_z(self, reg_z_var, nonreg_z_var):
        reg_z_vars = self.reg_latent_dist.split_var(reg_z_var)
        reg_idx = 0
        nonreg_z_vars = self.nonreg_latent_dist.split_var(nonreg_z_var)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_z_vars[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_z_vars[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_vars(ret)

    def combine_reg_nonreg_dist_info(self, reg_dist_info, nonreg_dist_info):
        reg_dist_infos = self.reg_latent_dist.split_dist_info(reg_dist_info)
        reg_idx = 0
        nonreg_dist_infos = self.nonreg_latent_dist.split_dist_info(nonreg_dist_info)
        nonreg_idx = 0
        ret = []
        for idx, (dist_i, reg_i) in enumerate(self.latent_spec):
            if reg_i:
                ret.append(reg_dist_infos[reg_idx])
                reg_idx += 1
            else:
                ret.append(nonreg_dist_infos[nonreg_idx])
                nonreg_idx += 1
        return self.latent_dist.join_dist_infos(ret)

    def add_images_to_summary(self, x_dist_flat, images_name,
                              collections=None, n_rows=10, n_columns=10):
        x_dist_info = self.output_dist.activate_dist(x_dist_flat)

        # just take the mean image
        if isinstance(self.output_dist, Bernoulli):
            img_var = x_dist_info["p"]
        elif isinstance(self.output_dist, Gaussian):
            img_var = x_dist_info["mean"]
        else:
            raise NotImplementedError
        img_var = self.dataset.inverse_transform(img_var)
        # (n, h, w, c)
        img_var = tf.reshape(img_var, [self.batch_size] + list(self.image_shape))
        img_var = img_var[:n_rows * n_columns, :, :, :]
        # (rows, rows, h, w, c)
        shape = [n_rows, n_columns] + list(self.image_shape)
        imgs = tf.reshape(img_var, shape)
        stacked_img = []
        for row in xrange(n_rows):
            row_img = []
            for col in xrange(n_columns):
                row_img.append(imgs[row, col, :, :, :])
            stacked_img.append(tf.concat(1, row_img))
        imgs = tf.concat(0, stacked_img)
        imgs = tf.expand_dims(imgs, 0)
        tf.summary.image(name=images_name, tensor=imgs,
                         collections=collections)

    def get_test_samples(self, sess, z_tensor, images_tensor, sampling_type,
                         collections=None, min_continuous=-2.,
                         max_continuous=2.):
        if collections is None:
            collections = ['samples']
        z_vars_and_names = make_list(self.get_z_var(sampling_type,
                                                    min_continuous,
                                                    max_continuous))
        for z_var, name in z_vars_and_names:
            feed_dict = {z_tensor.name: z_var}
            x_dist_flat = sess.run(images_tensor, feed_dict=feed_dict)
            self.add_images_to_summary(x_dist_flat, name,
                                       collections=collections)

    def get_train_samples(self, collections=None):
        if len(self.reg_latent_dist.dists) > 0:
            sampling_type = 'latent_code_influence'
        else:
            sampling_type = 'random'
        z_vars_and_names = make_list(self.get_z_var(sampling_type))
        for z_var, name in z_vars_and_names:
            _, _, x_dist_flat = self.generate(z_var)
            self.add_images_to_summary(x_dist_flat, name, collections)

    def get_z_var(self, sampling_type, min_continuous=-1., max_continuous=1.):
        """
        Args:
            sampling_type (str): The type of z_var to get
        """
        if sampling_type == 'random':
            return self.get_random_z_var()
        elif sampling_type == 'latent_code_influence':
            return self.get_latent_code_influence_z_var(
                min_continuous=min_continuous,
                max_continuous=max_continuous)
        elif sampling_type == 'linear_interpolation':
            return self.get_linear_interpolation_z_var()
        else:
            raise NotImplementedError

    def get_random_z_var(self):
        with tf.Session():
            # (n, d)
            z_var = self.latent_dist.sample_prior(self.batch_size).eval()
        return z_var, 'samples'

    def get_latent_code_influence_z_var(self, n_samples=10, n_variations=10,
                                        min_continuous=-1.,
                                        max_continuous=1.):
        """
        Args:
            n_samples (int): The number of different samples (n_columns).
            n_variations (int): The number of variations for each latent code
             (n_rows).
            min_continuous (float): The minimum value for a continuous latent
             code
            max_continuous (float): The maximum value for a continuous latent
             code
        """
        if len(self.reg_latent_dist.dists) == 0:
            raise ValueError('The model must have at least one regularization '
                             'latent distribution.')
        with tf.Session():
            # (n, d) with 10 * 10 samples + other samples
            n = n_samples * n_variations
            fixed_noncat = self.nonreg_latent_dist.sample_prior(n_samples)
            fixed_noncat = np.tile(fixed_noncat.eval(), [n_variations, 1])
            other = self.nonreg_latent_dist.sample_prior(self.batch_size - n)
            other = other.eval()
            fixed_noncat = np.concatenate([fixed_noncat, other], axis=0)

            fixed_cat = self.reg_latent_dist.sample_prior(n_samples).eval()
            fixed_cat = np.tile(fixed_cat, [n_variations, 1])
            other = self.reg_latent_dist.sample_prior(self.batch_size - n)
            other = other.eval()
            fixed_cat = np.concatenate([fixed_cat, other], axis=0)

        offset = 0
        z_vars_and_names = []
        for dist_idx, dist in enumerate(self.reg_latent_dist.dists):
            if isinstance(dist, Gaussian):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                vary_cat = dist.varying_values(min_continuous, max_continuous,
                                               n_variations)
                vary_cat = np.repeat(vary_cat, n_samples)
                other = np.zeros(self.batch_size - n)
                vary_cat = np.concatenate((vary_cat, other)).reshape((-1, 1))
                vary_cat = np.asarray(vary_cat, dtype=np.float32)

                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+1] = vary_cat
                offset += 1
            elif isinstance(dist, Categorical):
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(n_variations):
                    cat_ids.extend([idx] * n_samples)
                cat_ids.extend([0] * (self.batch_size - n))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+dist.dim] = lookup[cat_ids]
                offset += dist.dim
            elif isinstance(dist, Bernoulli):
                assert dist.dim == 1, "Only dim=1 is currently supported"
                lookup = np.eye(dist.dim, dtype=np.float32)
                cat_ids = []
                for idx in xrange(n_variations):
                    cat_ids.extend([int(idx / 5)] * n_samples)
                cat_ids.extend([0] * (self.batch_size - n))
                cur_cat = np.copy(fixed_cat)
                cur_cat[:, offset:offset+dist.dim] = np.expand_dims(np.array(cat_ids), axis=-1)
                # import ipdb; ipdb.set_trace()
                offset += dist.dim
            else:
                raise NotImplementedError
            # (n, d)
            # The 10 first rows have different z and c and are tiled 10 times
            # except for the varying c that is the same by blocks of 10 rows
            # and linearly varies between blocks
            z_var = np.concatenate([fixed_noncat, cur_cat], axis=1)

            # Images where each column had a different fixed z and c
            # The varying c varies along each column
            # (a different value for each row)
            name = 'image_{}_{}'.format(dist_idx, dist.__class__.__name__)
            z_vars_and_names.append((z_var, name))
        return z_vars_and_names

    def get_linear_interpolation_z_var(self, n_samples=10, n_variations=10):
        with tf.Session():
            n = n_samples * n_variations
            all_z_start = self.latent_dist.sample_prior(n_samples)
            all_z_end = self.latent_dist.sample_prior(n_samples)
            coefficients = np.linspace(start=0, stop=1, num=n_variations)
            z_var = []
            for z_start, z_end in zip(all_z_start.eval(), all_z_end.eval()):
                for coeff in coefficients:
                    z_var.append(coeff * z_start + (1 - coeff) * z_end)
            other = self.latent_dist.sample_prior(self.batch_size - n).eval()
            z_var = np.concatenate([z_var, other], axis=0)
            z_var = np.asarray(z_var, dtype=np.float32)
            return z_var, 'linear_interpolations'


class MNISTInfoGAN(RegularizedGAN):
    def build_network(self):
        with tf.variable_scope("d_net"):
            shared_template = \
                (pt.template("input").
                 reshape([-1] + list(self.image_shape)).
                 custom_conv2d(64, k_h=4, k_w=4).
                 apply(leaky_rectify).
                 custom_conv2d(128, k_h=4, k_w=4).
                 conv_batch_norm().
                 apply(leaky_rectify).
                 custom_fully_connected(1024).
                 fc_batch_norm().
                 apply(leaky_rectify))
            self.discriminator_template = shared_template.custom_fully_connected(
                1)
            self.encoder_template = \
                (shared_template.
                 custom_fully_connected(128).
                 fc_batch_norm().
                 apply(leaky_rectify).
                 custom_fully_connected(self.reg_latent_dist.dist_flat_dim))

        with tf.variable_scope("g_net"):
            self.generator_template = \
                (pt.template("input").
                 custom_fully_connected(1024).
                 fc_batch_norm().
                 apply(tf.nn.relu).
                 custom_fully_connected(self.image_size / 4 * self.image_size / 4 * 128).
                 fc_batch_norm().
                 apply(tf.nn.relu).
                 reshape([-1, self.image_size / 4, self.image_size / 4, 128]).
                 custom_deconv2d([0, self.image_size / 2, self.image_size / 2, 64],
                                 k_h=4, k_w=4).
                 conv_batch_norm().
                 apply(tf.nn.relu).
                 custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
                 flatten())


class CelebAInfoGAN(RegularizedGAN):
    def build_network(self):
        with tf.variable_scope("d_net"):
            shared_template = \
                (pt.template("input").
                 reshape([-1] + list(self.image_shape)).
                 custom_conv2d(64, k_h=4, k_w=4).
                 apply(leaky_rectify).
                 custom_conv2d(128, k_h=4, k_w=4).
                 conv_batch_norm().
                 apply(leaky_rectify).
                 custom_conv2d(256, k_h=4, k_w=4).
                 conv_batch_norm().
                 apply(leaky_rectify))
            self.discriminator_template = shared_template.custom_fully_connected(
                1)
            self.encoder_template = \
                (shared_template.
                 custom_fully_connected(128).
                 fc_batch_norm().
                 apply(leaky_rectify).
                 custom_fully_connected(self.reg_latent_dist.dist_flat_dim))

        with tf.variable_scope("g_net"):
            self.generator_template = \
                (pt.template("input").
                 custom_fully_connected(self.image_size / 16 * self.image_size / 16 * 448).
                 fc_batch_norm().
                 apply(tf.nn.relu).
                 reshape([-1, self.image_size / 16, self.image_size / 16, 448]).
                 custom_deconv2d([0, self.image_size / 8, self.image_size / 8, 256],
                                 k_h=4, k_w=4).
                 conv_batch_norm().
                 apply(tf.nn.relu).
                 custom_deconv2d([0, self.image_size / 4, self.image_size / 4, 128],
                                 k_h=4, k_w=4).
                 apply(tf.nn.relu).
                 custom_deconv2d([0, self.image_size / 2, self.image_size / 2, 64],
                                 k_h=4, k_w=4).
                 apply(tf.nn.relu).
                 custom_deconv2d([0] + list(self.image_shape), k_h=4, k_w=4).
                 apply(tf.nn.tanh).
                 flatten())


class CIFAR10RegularizedGAN(RegularizedGAN):
    def build_network(self):
        # TODO: Check leaky RELU leak rate
        # TODO: Binary vs 10-class discriminator
        with tf.variable_scope("d_net"):
            shared_template = \
                (pt.template("input").
                 reshape([-1] + list(self.image_shape)).
                 custom_conv2d(96, k_h=3, k_w=3, d_h=1, d_w=1).
                 apply(leaky_rectify).
                 custom_conv2d(96, k_h=3, k_w=3, d_h=1, d_w=1).
                 apply(leaky_rectify).
                 custom_conv2d(96, k_h=3, k_w=3, d_h=1, d_w=1).
                 apply(leaky_rectify).
                 max_pool(kernel=[2, 2], stride=[2, 2]).
                 custom_conv2d(192, k_h=3, k_w=3, d_h=1, d_w=1).
                 apply(leaky_rectify).
                 custom_conv2d(192, k_h=3, k_w=3, d_h=1, d_w=1).
                 apply(leaky_rectify).
                 custom_conv2d(192, k_h=3, k_w=3, d_h=1, d_w=1).
                 apply(leaky_rectify).
                 max_pool(kernel=[3, 3], stride=[2, 2]).
                 custom_conv2d(192, k_h=3, k_w=3, d_h=1, d_w=1).
                 apply(leaky_rectify).
                 custom_conv2d(192, k_h=1, k_w=1, d_h=1, d_w=1).
                 apply(leaky_rectify).
                 # custom_conv2d(10, k_h=1, k_w=1, d_h=1, d_w=1).
                 # apply(leaky_rectify).
                 custom_fully_connected(1024).
                 fc_batch_norm().
                 apply(leaky_rectify))
            self.discriminator_template = shared_template.custom_fully_connected(
                1)
            self.encoder_template = \
                (shared_template.
                 custom_fully_connected(128).
                 fc_batch_norm().
                 apply(leaky_rectify).
                 custom_fully_connected(self.reg_latent_dist.dist_flat_dim))

        # TODO: use perforated upsampling
        with tf.variable_scope("g_net"):
            self.generator_template = \
                (pt.template("input").
                 custom_fully_connected(self.image_size / 4 * self.image_size / 4 * 192).
                 apply(leaky_rectify).
                 reshape([-1, self.image_size / 4, self.image_size / 4, 192]).
                 apply(tf.image.resize_nearest_neighbor, [self.image_size / 2, self.image_size / 2]).
                 custom_conv2d(96, k_h=5, k_w=5, d_h=1, d_w=1).
                 apply(leaky_rectify).
                 custom_conv2d(96, k_h=5, k_w=5, d_h=1, d_w=1).
                 apply(leaky_rectify).
                 apply(tf.image.resize_nearest_neighbor, [self.image_size, self.image_size]).
                 custom_conv2d(96, k_h=5, k_w=5, d_h=1, d_w=1).
                 apply(leaky_rectify).
                 custom_conv2d(3, k_h=5, k_w=5, d_h=1, d_w=1).  # BUG
                 apply(leaky_rectify).  # BUG
                 flatten())


class LSUN_DCGAN(RegularizedGAN):
    def build_network(self):
        with tf.variable_scope("d_net"):
            shared_template = \
                (pt.template("input").
                 reshape([-1] + list(self.image_shape)).
                 custom_conv2d(128, k_h=5, k_w=5).
                 conv_batch_norm().
                 apply(leaky_rectify).
                 custom_conv2d(256, k_h=5, k_w=5).
                 conv_batch_norm().
                 apply(leaky_rectify).
                 custom_conv2d(512, k_h=5, k_w=5).
                 conv_batch_norm().
                 apply(leaky_rectify))
            self.discriminator_template = shared_template.custom_fully_connected(
                1)
            self.encoder_template = \
                (shared_template.
                 custom_fully_connected(128).
                 fc_batch_norm().
                 apply(leaky_rectify).
                 custom_fully_connected(self.reg_latent_dist.dist_flat_dim))

        with tf.variable_scope("g_net"):
            self.generator_template = \
                (pt.template("input").
                 custom_fully_connected(512 * self.image_size / 8 * self.image_size / 8).
                 fc_batch_norm().
                 apply(tf.nn.relu).
                 reshape([-1, 512, self.image_size / 8, self.image_size / 8]).
                 custom_deconv2d([0, self.image_size / 4, self.image_size / 4, 256],
                                 k_h=5, k_w=5).
                 conv_batch_norm().
                 apply(tf.nn.relu).
                 custom_deconv2d([0, self.image_size / 2, self.image_size / 2, 128],
                                 k_h=5, k_w=5).
                 conv_batch_norm().
                 apply(tf.nn.relu).
                 custom_deconv2d([0] + list(self.image_shape),
                                 k_h=5, k_w=5).
                 apply(tf.nn.tanh).
                 flatten())


class CelebA_DCGAN_carpedm20(RegularizedGAN):
    def build_network(self):
        with tf.variable_scope("d_net"):
            shared_template = \
                (pt.template("input").
                 reshape([-1] + list(self.image_shape)).
                 custom_conv2d(64, k_h=5, k_w=5).
                 conv_batch_norm().
                 apply(leaky_rectify).
                 custom_conv2d(128, k_h=5, k_w=5).
                 conv_batch_norm().
                 apply(leaky_rectify).
                 custom_conv2d(256, k_h=5, k_w=5).
                 conv_batch_norm().
                 apply(leaky_rectify).
                 custom_conv2d(512, k_h=5, k_w=5).
                 conv_batch_norm().
                 apply(leaky_rectify))
            self.discriminator_template = shared_template.custom_fully_connected(
                1)
            self.encoder_template = \
                (shared_template.
                 custom_fully_connected(128).
                 fc_batch_norm().
                 apply(leaky_rectify).
                 custom_fully_connected(self.reg_latent_dist.dist_flat_dim))

        with tf.variable_scope("g_net"):
            self.generator_template = \
                (pt.template("input").
                 custom_fully_connected(512 * self.image_size / 8 * self.image_size / 8).
                 fc_batch_norm().
                 apply(tf.nn.relu).
                 reshape([-1, 512, self.image_size / 8, self.image_size / 8]).
                 custom_deconv2d([0, self.image_size / 4, self.image_size / 4, 256],
                                 k_h=5, k_w=5).
                 conv_batch_norm().
                 apply(tf.nn.relu).
                 custom_deconv2d([0, self.image_size / 2, self.image_size / 2, 128],
                                 k_h=5, k_w=5).
                 conv_batch_norm().
                 apply(tf.nn.relu).
                 custom_deconv2d([0] + list(self.image_shape),
                                 k_h=5, k_w=5).
                 apply(tf.nn.tanh).
                 flatten())
