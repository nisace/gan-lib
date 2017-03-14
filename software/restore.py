import numpy as np
import tensorflow as tf

if __name__ == '__main__':
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph('ckt/test/test.ckpt.meta')
        new_saver.restore(sess, 'ckt/test/test.ckpt')
        loaded_y = tf.get_collection("saved_y")[0]
        loaded_x = tf.get_collection("saved_x")[0]
        print(sess.run(loaded_y, {loaded_x: [0, 10, 20]}))
