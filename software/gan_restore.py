import numpy as np
import tensorflow as tf
import os
import shutil

if __name__ == '__main__':
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(
            'ckt/mnist_infogan/mnist_infogan_2017_03_14_12_03_25/mnist_infogan_2017_03_14_12_03_25_500.ckpt.meta')
        new_saver.restore(sess,
                          'ckt/mnist_infogan/mnist_infogan_2017_03_14_12_03_25/mnist_infogan_2017_03_14_12_03_25_500.ckpt')

        # loaded_z = tf.get_collection("z_var")[0]
        # print(loaded_z)
        # tf.assign(loaded_z, np.random.rand(3))
        # loaded_img = tf.get_collection("images")[0]

        loaded_img = tf.get_collection("generated")[0]

        # imgs = sess.run(loaded_img)
        # tf.summary.image(name='loaded_images', tensor=imgs)

        img = sess.run(loaded_img)
        print(img)
        print(img.shape)

        img = img[0, :].reshape(1, 28, 28, 1)

        folder = "logs/test"
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        summary_writer = tf.summary.FileWriter(folder, sess.graph)
        sum = tf.summary.image(name='loaded_images', tensor=img)
        summary_op = tf.summary.merge([sum])
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, 0)
