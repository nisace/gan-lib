import prettytensor as pt
import tensorflow as tf

from infogan.misc.custom_ops import leaky_rectify, custom_conv2d
from infogan.models.regularized_gan import GANModel


class Data2DataGAN(GANModel):
    def __init__(self, input_dataset, **kwargs):
        self.input_dataset = input_dataset
        self.input_shape = input_dataset.image_shape
        self.input_size = self.input_shape[0]
        super(Data2DataGAN, self).__init__(**kwargs)

    def build_g_input(self):
        g_input_shape = [self.batch_size, self.input_dataset.image_dim]
        return tf.placeholder(tf.float32, g_input_shape)

    # def g_input(self, batch_size=None):
    #     if batch_size is None:
    #         batch_size = self.batch_size
    #     d_input_shape = [batch_size, self.input_dataset.image_dim]
    #     return tf.placeholder(tf.float32, d_input_shape)

    def get_g_feed_dict(self):
        x, _ = self.input_dataset.train.next_batch(self.batch_size)
        return {self.g_input: x}

    def modify_summary_images(self, images):
        shape = [1] + list(self.input_shape)
        shape[1] *= self.batch_size
        input_images = tf.reshape(self.g_input, shape)
        return tf.concat(2, [input_images, images])


class Horse2Zebra_CycleGAN(Data2DataGAN):
    def build_network(self, scope_suffix=''):
        with tf.variable_scope("d_net{}".format(scope_suffix)):
            paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
            mode = "CONSTANT"
            self.discriminator_template = \
                (pt.template("input").
                 reshape([-1] + list(self.output_shape)).
                 custom_conv2d(64, k_h=4, k_w=4, d_h=2, d_w=2).
                 apply(leaky_rectify, 0.2).
                 custom_conv2d(128, k_h=4, k_w=4, d_h=2, d_w=2).
                 conv_instance_norm().
                 apply(leaky_rectify, 0.2).
                 custom_conv2d(256, k_h=4, k_w=4, d_h=2, d_w=2).
                 conv_instance_norm().
                 apply(leaky_rectify, 0.2).
                 apply(tf.pad, paddings, mode).
                 custom_conv2d(512, k_h=4, k_w=4, d_h=2, d_w=2, padding='VALID').
                 conv_instance_norm().
                 apply(leaky_rectify, 0.2).
                 apply(tf.pad, paddings, mode).
                 custom_conv2d(1, k_h=4, k_w=4, d_h=1, d_w=1, padding='VALID'))

        with tf.variable_scope("g_net{}".format(scope_suffix)):
            # TODO: Add reflection padding
            # with apply(tf.pad([[?,?], [?,?]], 'REFLECT'))
            # and padding='VALID' in custom_conv_2d
            paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]
            mode = "REFLECT"
            self.generator_template = \
                (pt.template("input").
                 reshape([-1] + list(self.input_shape)).
                 apply(tf.pad, [[0, 0], [3, 3], [3, 3], [0, 0]], mode).
                 custom_conv2d(32, k_h=7, k_w=7, d_h=1, d_w=1, padding='VALID').
                 conv_instance_norm().
                 apply(tf.nn.relu).
                 apply(tf.pad, paddings, mode).
                 custom_conv2d(64, k_h=3, k_w=3, d_h=2, d_w=2, padding='VALID').
                 conv_instance_norm().
                 apply(tf.nn.relu).
                 apply(tf.pad, paddings, mode).
                 custom_conv2d(128, k_h=3, k_w=3, d_h=2, d_w=2, padding='VALID').
                 conv_instance_norm().
                 apply(tf.nn.relu).
                 custom_residual(paddings, mode, custom_conv2d, 128, k_h=3, k_w=3, d_h=1, d_w=1, padding='VALID').
                 custom_residual(paddings, mode, custom_conv2d, 128, k_h=3, k_w=3, d_h=1, d_w=1, padding='VALID').
                 custom_residual(paddings, mode, custom_conv2d, 128, k_h=3, k_w=3, d_h=1, d_w=1, padding='VALID').
                 custom_residual(paddings, mode, custom_conv2d, 128, k_h=3, k_w=3, d_h=1, d_w=1, padding='VALID').
                 custom_residual(paddings, mode, custom_conv2d, 128, k_h=3, k_w=3, d_h=1, d_w=1, padding='VALID').
                 custom_residual(paddings, mode, custom_conv2d, 128, k_h=3, k_w=3, d_h=1, d_w=1, padding='VALID').
                 custom_residual(paddings, mode, custom_conv2d, 128, k_h=3, k_w=3, d_h=1, d_w=1, padding='VALID').
                 custom_residual(paddings, mode, custom_conv2d, 128, k_h=3, k_w=3, d_h=1, d_w=1, padding='VALID').
                 custom_residual(paddings, mode, custom_conv2d, 128, k_h=3, k_w=3, d_h=1, d_w=1, padding='VALID').
                 custom_deconv2d([0, self.output_size / 2, self.output_size / 2, 64],
                                 k_h=3, k_w=3, d_h=2, d_w=2,  padding='SAME'). # 'VALID' or 'SAME' ?
                 conv_instance_norm().
                 apply(tf.nn.relu).
                 custom_deconv2d([0, self.output_size, self.output_size, 32],
                                 k_h=3, k_w=3, d_h=2, d_w=2,  padding='SAME'). # 'VALID' or 'SAME' ?
                 conv_instance_norm().
                 apply(tf.nn.relu).
                 apply(tf.pad, [[0, 0], [3, 3], [3, 3], [0, 0]], mode).
                 custom_conv2d(3, k_h=7, k_w=7, d_h=1, d_w=1, padding='VALID').
                 # conv_instance_norm().
                 # apply(tf.nn.relu).
                 apply(tf.nn.tanh).
                 flatten())
