import tensorflow as tf
import prettytensor as pt
from infogan.algos.loss_builder2 import AbstractLossBuilder


class CycleGANLossBuilder(AbstractLossBuilder):
    def __init__(self, loss_builders, cycle_loss_factor=10.0, **kwargs):
        """
        Args:
            models (list): List of 2 models [A->B, B->A]
        """
        self.loss_builders = loss_builders
        self.cycle_loss_factor = cycle_loss_factor
        models = [l.model for l in loss_builders]
        super(CycleGANLossBuilder, self).__init__(models=models, **kwargs)

    def get_g_feed_dict(self):
        feed_dict = self.loss_builders[0].get_g_feed_dict()
        feed_dict.update(self.loss_builders[1].get_g_feed_dict())
        return feed_dict

    def get_d_feed_dict(self):
        feed_dict = self.loss_builders[0].get_d_feed_dict()
        feed_dict[self.loss_builders[1].model.g_input] = feed_dict[
            self.loss_builders[0].model.d_input]
        feed_dict[self.loss_builders[1].model.d_input] = feed_dict[
            self.loss_builders[0].model.g_input]
        return feed_dict

    def init_loss(self):
        # print("\n".join(["Trainable variables"] + [v.name for v in tf.trainable_variables()]))
        # Initialize losses
        for i, loss_builder in enumerate(self.loss_builders):
            print('Build loss builder %s' % i)
            loss_builder.init_loss()
            # print("\n".join(["Trainable variables"] + [v.name for v in tf.trainable_variables()]))

        # Sum generators losses
        self.g_loss = tf.add_n([l.g_loss for l in self.loss_builders])

        # Add cycle consistency loss
        def get_sub_cycle_loss(model_1, model_2):
            print("model_1")
            fake_x_1 = model_1.generate()
            print("model_2")
            with pt.defaults_scope(phase=pt.Phase.train):
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    # scope.reuse_variables()
                    fake_x_2 = model_2.generate(fake_x_1)
            print("model_3")
            x_2 = model_1.g_input
            return tf.reduce_sum(tf.abs(fake_x_2 - x_2))

        print('Cycle loss 0')
        cycle_loss = get_sub_cycle_loss(self.loss_builders[0].model,
                                        self.loss_builders[1].model)
        print('Cycle loss 1')
        cycle_loss += get_sub_cycle_loss(self.loss_builders[1].model,
                                         self.loss_builders[0].model)
        self.g_loss += self.cycle_loss_factor * cycle_loss

        # Sum discriminators losses
        self.d_loss = tf.add_n([l.d_loss for l in self.loss_builders])

    def add_train_samples_to_summary(self):
        pass
