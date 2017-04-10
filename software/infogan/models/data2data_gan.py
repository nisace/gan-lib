import prettytensor as pt
import tensorflow as tf

from infogan.misc.custom_ops import leaky_rectify
from infogan.models.regularized_gan import GANModel


class Data2DataGAN(GANModel):
    def __init__(self, input_dataset, **kwargs):
        super(Data2DataGAN, self).__init__(**kwargs)
        self.input_dataset = input_dataset
        self.input_shape = input_dataset.image_shape
        self.input_size = self.input_shape[0]

    def g_input(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        d_input_shape = [batch_size, self.input_dataset.image_dim]
        return tf.placeholder(tf.float32, d_input_shape)


class Horse2Zebra_CycleGAN(Data2DataGAN):
    def build_network(self):
        with tf.variable_scope("d_net"):
            shared_template = \
                (pt.template("input").
                 reshape([-1] + list(self.output_shape)).
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
            # TODO: Add reflection padding
            # with apply(tf.pad([[?,?], [?,?]], 'REFLECT'))
            # and padding='VALID' in custom_conv_2d
            self.generator_template = \
                (pt.template("input").
                 reshape([-1] + list(self.input_shape)).
                 custom_conv2d(32, k_h=7, k_w=7, d_h=1, d_w=1).
                 conv_instance_norm().
                 apply(leaky_rectify, 0.2).
                 custom_conv2d(64, k_h=3, k_w=3, d_h=2, d_w=2).
                 conv_instance_norm().
                 apply(leaky_rectify, 0.2).
                 custom_conv2d(128, k_h=3, k_w=3, d_h=2, d_w=2).
                 conv_instance_norm().
                 apply(leaky_rectify, 0.2).


                 custom_fully_connected(self.output_size / 16 * self.output_size / 16 * 448).
                 fc_batch_norm().
                 apply(tf.nn.relu).
                 reshape([-1, self.output_size / 16, self.output_size / 16, 448]).
                 custom_deconv2d([0, self.output_size / 8, self.output_size / 8, 256],
                                 k_h=4, k_w=4).
                 conv_batch_norm().
                 apply(tf.nn.relu).
                 custom_deconv2d([0, self.output_size / 4, self.output_size / 4, 128],
                                 k_h=4, k_w=4).
                 apply(tf.nn.relu).
                 custom_deconv2d([0, self.output_size / 2, self.output_size / 2, 64],
                                 k_h=4, k_w=4).
                 apply(tf.nn.relu).
                 custom_deconv2d([0] + list(self.output_shape), k_h=4, k_w=4).
                 apply(tf.nn.tanh).
                 flatten())

