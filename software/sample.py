import os

import numpy as np
import tensorflow as tf

from utils.date_time_utils import get_timestamp


def sample(checkpoint_path, image_shape, model, sampling_type, z_value=None, n_rows=1, n_columns=1):
    """
    Args:
        checkpoint_path (str): The checkpoint file path (.ckpt file)
        image_shape (tuple): The image shape is the format (h, w, c)
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
        images_tensor = tf.get_collection("x_dist_flat")[0]
        z_tensor = tf.get_collection("z_var")[0]
        collection = 'samples'
        model.get_samples_test(sess, z_tensor, images_tensor, sampling_type,
                               collections=[collection])


        # feed_dict = None
        # if z_value is not None:
        #     if len(z_value.shape) != 2:
        #         raise ValueError("z_value must be a 2d array")
        #     print("Getting input noise tensor")
        #     z_tensor = tf.get_collection("z_var")[0]
        #     z_tensor_shape = z_tensor.get_shape().as_list()
        #     if z_value.shape[1] != z_tensor_shape[1]:
        #         msg = 'z_value.shape[1] [{}] must be equal to ' \
        #               'z_tensor.shape[1] [{}]'
        #         msg = msg.format(z_value.shape[1], z_tensor_shape[1])
        #         raise ValueError(msg)
        #     z = np.ones(z_tensor_shape)
        #     z[:z_value.shape[0], :] = z_value
        #     feed_dict[z_tensor.name] = z
        #
        # print("Sampling")
        # # (batch_size, h * w * c)
        # images = sess.run(images_tensor, feed_dict=feed_dict)
        # # images = model.output_dist.activate_dist(images)
        # # images = images[:n_rows * n_columns, :]
        # # images = images.reshape(n_rows, n_columns, *image_shape)
        # # stacked_img = []
        # # for row in xrange(n_rows):
        # #     row_img = []
        # #     for col in xrange(n_columns):
        # #         row_img.append(images[row, col, :, :, :])
        # #     stacked_img.append(np.concatenate(1, row_img))
        # # imgs = tf.concat(0, stacked_img)
        # # imgs = tf.expand_dims(imgs, 0)

        samples_folder = os.path.relpath(checkpoint_path, 'ckt')
        samples_folder = os.path.dirname(samples_folder)
        name = 'samples_{}'.format(get_timestamp())
        samples_folder = os.path.join('logs', samples_folder, name)
        summary_writer = tf.summary.FileWriter(samples_folder, sess.graph)

        # sum = model.add_images_to_summary(images, 'samples')
        # sum = tf.summary.image(name='samples', tensor=images)

        summary_op = tf.summary.merge_all(key=collection)
        # summary_op = tf.summary.merge([sum])

        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, 0)
        print("Samples saved in {}".format(samples_folder))
