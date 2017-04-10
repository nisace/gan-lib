import tensorflow as tf

from infogan.algos.loss_builder2 import AbstractLossBuilder


class CycleGANLossBuilder(AbstractLossBuilder):
    def __init__(self, loss_builders, cycle_loss_factor=10.0):
        """
        Args:
            models (list): List of 2 models [A->B, B->A]
        """
        self.loss_builders = loss_builders
        self.cycle_loss_factor = cycle_loss_factor
        super(CycleGANLossBuilder, self).__init__(models=models)

    def get_g_feed_dict(self):
        feed_dict = self.get_d_feed_dict()
        for loss_builder in self.loss_builders:
            del feed_dict[loss_builder.d_input()]
        return feed_dict

    def get_d_feed_dict(self):
        feed_dict = None
        for loss_builder in self.loss_builders:
            d = loss_builder.get_g_feed_dict()
            if d is None:
                continue
            feed_dict = d.copy() if feed_dict is None else feed_dict.update(d)
        feed_dict[self.loss_builders[0].model.g_input()] = feed_dict[
            self.loss_builders[1].model.d_input()]
        feed_dict[self.loss_builders[1].model.g_input()] = feed_dict[
            self.loss_builders[0].model.d_input()]
        return feed_dict

    def init_loss(self):
        # Initialize losses
        for loss_builder in self.loss_builders:
            loss_builder.init_loss()

        # Sum generators losses
        self.g_loss = tf.add_n([l.generator_loss for l in self.loss_builders])

        # Add cycle consistency loss
        def get_sub_cycle_loss(model_1, model_2):
            fake_x_1 = model_1.generate()
            fake_x_2 = model_2.generate(fake_x_1)
            x_2 = model_1.g_input()
            return tf.reduce_sum(tf.abs(fake_x_2 - x_2))

        cycle_loss = get_sub_cycle_loss(self.loss_builders[0].model,
                                        self.loss_builders[1].model)
        cycle_loss += get_sub_cycle_loss(self.loss_builders[1].model,
                                         self.loss_builders[0].model)
        self.g_loss += self.cycle_loss_factor * cycle_loss

        # Sum discriminators losses
        self.d_loss = tf.add_n([l.discrim_loss for l in self.loss_builders])

    def add_train_samples_to_summary(self):
        pass
