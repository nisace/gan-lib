import os
import shutil

import numpy as np
import tensorflow as tf


def run():
    shape = 3

    # Create 100 phony x, y data points in NumPy, y = x * 0.1 + 0.3
    x_data = tf.placeholder(tf.float32, shape=(shape,))
    # x_data = np.random.rand(100).astype(np.float32)
    v = 10 * x_data
    y_data = v * 0.1 + 0.3
    # y_data = x_data * 0.1 + 0.3

    # Try to find values for W and b that compute y_data = W * x_data + b
    # (We know that W should be 0.1 and b 0.3, but TensorFlow will
    # figure that out for us.)
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W * x_data + b

    tf.add_to_collection("saved_y", y)
    tf.add_to_collection("saved_x", x_data)

    # Minimize the mean squared errors.
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)

    # Before starting, initialize the variables.  We will 'run' this first.
    init = tf.global_variables_initializer()

    # Launch the graph.
    sess = tf.Session()
    sess.run(init)

    saver = tf.train.Saver()

    # Fit the line.
    for step in range(201):
        sess.run(train, {x_data: np.random.rand(shape)})
        if step % 20 == 0:
            print(step, sess.run(W), sess.run(b))

            # Learns best fit is W: [0.1], b: [0.3]
    folder = "ckt/test"
    if os.path.exists(folder):
        shutil.rmtree("ckt/test")
    os.makedirs(folder)
    saver.save(sess, "ckt/test/test.ckpt")


def restore():
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('ckt/test/test.ckpt.meta')
        new_saver.restore(sess, 'ckt/test/test.ckpt')
        loaded_y = tf.get_collection("saved_y")[0]
        loaded_x = tf.get_collection("saved_x")[0]
        print(sess.run(loaded_y, {loaded_x: [0, 10, 20]}))
