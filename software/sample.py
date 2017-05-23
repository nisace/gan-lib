import os

import numpy as np
import tensorflow as tf

from utils.date_time_utils import get_timestamp


def sample(checkpoint_path, model, sampling_type, n_samples=1):
    """
    Args:
        checkpoint_path (str): The checkpoint file path (.ckpt file).
        sampling_type (str): The type of sampling to perform.
        model (RegularizedGAN): The GAN model object.
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

        print("Sampling")
        with tf.variable_scope("samples", reuse=True):
            for _ in range(n_samples):
                model.get_test_samples(sess, z_tensor, images_tensor,
                                       sampling_type, collections=[collection])

        print("Writing summary")
        # Get summary_writer
        samples_folder = os.path.relpath(checkpoint_path, 'ckt')
        samples_folder = os.path.dirname(samples_folder)
        name = 'samples_{}'.format(get_timestamp())
        samples_folder = os.path.join('logs', samples_folder, name)
        summary_writer = tf.summary.FileWriter(samples_folder, sess.graph)

        # Write summary
        summary_op = tf.summary.merge_all(key=collection)
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, 0)
        print("Samples saved in {}".format(samples_folder))
