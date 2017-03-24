import os

import numpy as np
import tensorflow as tf

from utils.date_time_utils import get_timestamp


def sample(checkpoint_path, z_value=None, n_rows=1, n_columns=1):
    """
    Args:
        checkpoint_path (str): The checkpoint file path (.ckpt file)
        z_value (ndarray, default None): The value of the noise variable z.
         If None, the value is sampled from the variable prior.
        n_rows (int): The number of samples rows.
        n_columns (int): The number of samples columns.
    """
    if checkpoint_path[-5:] == '.meta':
        checkpoint_path = checkpoint_path[:-5]
    with tf.Session() as sess:
        print("Importing meta graph")
        saver = tf.train.import_meta_graph(checkpoint_path + '.meta')
        saver.restore(sess, checkpoint_path)
        print("Getting samples tensor")
        images_tensor = tf.get_collection("generated")[0]

        feed_dict = None
        if z_value is not None:
            if len(z_value.shape) != 2:
                raise ValueError("z_value must be a 2d array")
            print("Getting input noise tensor")
            z_tensor = tf.get_collection("z_var")[0]
            z_tensor_shape = z_tensor.get_shape().as_list()
            if z_value.shape[1] != z_tensor_shape[1]:
                msg = 'z_value.shape[1] [{}] must be equal to ' \
                      'z_tensor.shape[1] [{}]'
                msg = msg.format(z_value.shape[1], z_tensor_shape[1])
                raise ValueError(msg)
            z = np.ones(z_tensor_shape)
            z[:z_value.shape[0], :] = z_value
            feed_dict[z_tensor.name] = z

        print("Sampling")
        # (batch_size, h * w * c)
        images = sess.run(images_tensor, feed_dict=feed_dict)
        images = images[0, :].reshape(1, 28, 28, 1)

        samples_folder = os.path.relpath(checkpoint_path, 'ckt')
        samples_folder = os.path.dirname(samples_folder)
        name = 'samples_{}'.format(get_timestamp())
        samples_folder = os.path.join('logs', samples_folder, name)

        summary_writer = tf.summary.FileWriter(samples_folder, sess.graph)
        sum = tf.summary.image(name='samples', tensor=images)
        summary_op = tf.summary.merge([sum])
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, 0)
        print("Samples saved in {}".format(samples_folder))
