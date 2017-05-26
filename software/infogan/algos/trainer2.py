import cPickle as pkl
import os
import sys

import numpy as np
import tensorflow as tf
from progressbar import ETA, Bar, Percentage, ProgressBar

from infogan.models.regularized_gan import RegularizedGAN

TINY = 1e-8


def apply_optimizer(optimizer, losses, var_list, clip_by_value=None,
                    name='cost'):
    total_loss = tf.add_n(losses, name=name)
    grads_and_vars = optimizer.compute_gradients(total_loss, var_list=var_list)
    if clip_by_value is not None:
        clipped_grads_and_vars = []
        for g, v in grads_and_vars:
            cg = None
            if g is not None:
                cg = tf.clip_by_value(g, *clip_by_value)
            clipped_grads_and_vars.append((cg, v))
        grads_and_vars = clipped_grads_and_vars
    train_op = optimizer.apply_gradients(grads_and_vars)
    return train_op


class Trainer(object):
    def __init__(self,
                 loss_builder,
                 exp_name="experiment",
                 log_dir="logs",
                 checkpoint_dir="ckt",
                 max_epoch=100,
                 updates_per_epoch=100,
                 snapshot_interval=500,
                 ):
        """
        :type model: RegularizedGAN
        """
        self.loss_builder = loss_builder
        self.max_epoch = max_epoch
        self.exp_name = exp_name
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.snapshot_interval = snapshot_interval
        self.updates_per_epoch = updates_per_epoch

    def update(self, sess, i, log_vars, all_log_vals):
        raise NotImplementedError

    def train(self):
        model_path = os.path.join(self.checkpoint_dir, 'models.pkl')
        with open(model_path, 'wb') as f:
            pkl.dump(self.loss_builder.models, f)

        # for i, model in enumerate(self.loss_builder.models):
        #     name = 'model_{}.pkl'.format(i)
        #     model_path = os.path.join(self.checkpoint_dir, name)
        #     with open(model_path, 'wb') as f:
        #         pkl.dump(model, f)

        self.loss_builder.init_opt()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.log_dir, sess.graph)

            saver = tf.train.Saver()

            counter = 0

            log_vars = [x for _, x in self.loss_builder.log_vars]
            log_keys = [x for x, _ in self.loss_builder.log_vars]

            for epoch in range(self.max_epoch):
                widgets = ["epoch #%d|" % epoch, Percentage(), Bar(), ETA()]
                pbar = ProgressBar(maxval=self.updates_per_epoch, widgets=widgets)
                pbar.start()

                all_log_vals = []
                for i in range(self.updates_per_epoch):
                    pbar.update(i)
                    all_log_vals = self.update(sess, i, log_vars, all_log_vals)
                    counter += 1

                    if counter % self.snapshot_interval == 0:
                        snapshot_name = "%s_%s" % (self.exp_name, str(counter))
                        fn = saver.save(sess, "%s/%s.ckpt" % (self.checkpoint_dir, snapshot_name))
                        print("Model saved in file: %s" % fn)

                # (n, h, w, c)
                feed_dict = self.loss_builder.get_d_feed_dict()
                summary_str = sess.run(summary_op, feed_dict)

                summary_writer.add_summary(summary_str, counter)

                avg_log_vals = np.mean(np.array(all_log_vals), axis=0)
                log_dict = dict(zip(log_keys, avg_log_vals))

                log_line = "; ".join("%s: %s" % (str(k), str(v)) for k, v in zip(log_keys, avg_log_vals))
                print("Epoch %d | " % (epoch) + log_line)
                sys.stdout.flush()
                if np.any(np.isnan(avg_log_vals)):
                    raise ValueError("NaN detected!")


class GANTrainer(Trainer):
    def __init__(self, **kwargs):
        self.gen_disc_update_ratio = 1
        super(GANTrainer, self).__init__(**kwargs)

    def update(self, sess, i, log_vars, all_log_vals):
        feed_dict = self.loss_builder.get_g_feed_dict()
        sess.run(self.loss_builder.g_trainer, feed_dict)
        if i % self.gen_disc_update_ratio == 0:
            feed_dict = self.loss_builder.get_d_feed_dict()
            log_vals = sess.run([self.loss_builder.d_trainer] + log_vars, feed_dict)[1:]
            all_log_vals.append(log_vals)
        return all_log_vals


class WassersteinGANTrainer(Trainer):
    def __init__(self, **kwargs):
        self.n_critic = 5
        self.d_weight_clip_by_value = [-0.01, 0.01]
        self.clip = None
        super(WassersteinGANTrainer, self).__init__(**kwargs)

    def update(self, sess, i, log_vars, all_log_vals):
        for _ in range(self.n_critic):
            feed_dict = self.loss_builder.get_d_feed_dict()
            log_vals = sess.run([self.loss_builder.d_trainer] + log_vars,
                                feed_dict)[1:]
            if self.clip is None:
                self.clip = [tf.assign(
                    d, tf.clip_by_value(d, *self.d_weight_clip_by_value))
                             for d in self.loss_builder.d_vars]
            sess.run(self.clip)
            all_log_vals.append(log_vals)
        feed_dict = self.loss_builder.get_g_feed_dict()
        sess.run(self.loss_builder.g_trainer, feed_dict)
        return all_log_vals
