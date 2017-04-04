import tensorflow as tf

TINY = 1e-8


class Loss(object):
    @staticmethod
    def get_d_loss(real_d, fake_d):
        """ Returns the discriminator loss tensor. """
        raise NotImplementedError

    @staticmethod
    def get_g_loss(fake_d):
        """ Returns the generator loss tensor. """
        raise NotImplementedError


class GANLoss(Loss):
    @staticmethod
    def get_d_loss(real_d, fake_d):
        real = tf.log(real_d + TINY)
        fake = tf.log(1. - fake_d + TINY)
        return - tf.reduce_mean(real + fake)

    @staticmethod
    def get_g_loss(fake_d):
        return - tf.reduce_mean(tf.log(fake_d + TINY))


class WassersteinGANLoss(Loss):
    @staticmethod
    def get_d_loss(real_d, fake_d):
        return tf.reduce_mean(real_d - fake_d)

    @staticmethod
    def get_g_loss(fake_d):
        return tf.reduce_mean(fake_d)
