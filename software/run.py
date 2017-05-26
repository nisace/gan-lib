from __future__ import absolute_import
from __future__ import print_function

import os

import tensorflow as tf

from infogan.algos import trainer2
from infogan.algos.cycle_gan_loss_builder import CycleGANLossBuilder
from infogan.algos.loss import GANLoss, LeastSquaresGANLoss, WassersteinGANLoss
from infogan.algos.loss_builder2 import InfoGANLossBuilder, GANLossBuilder
from infogan.misc.datasets import MnistDataset, CelebADataset, \
    HorseOrZebraDataset
from infogan.misc.distributions import Uniform, Categorical, MeanBernoulli, \
    MeanGaussian
from infogan.models.data2data_gan import Horse2Zebra_CycleGAN
from infogan.models.regularized_gan import MNISTInfoGAN, \
    CelebAInfoGAN
from utils.date_time_utils import get_timestamp
from utils.file_system_utils import make_exists


def train(model_name, learning_params):
    timestamp = get_timestamp()

    root_log_dir = os.path.join('logs', model_name)
    root_checkpoint_dir = os.path.join('ckt', model_name)
    experiment_name = '{}_{}'.format(model_name, timestamp)
    log_dir = os.path.join(root_log_dir, experiment_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, experiment_name)
    make_exists(log_dir)
    make_exists(checkpoint_dir)

    batch_size = learning_params['batch_size']
    updates_per_epoch = learning_params['updates_per_epoch']
    max_epoch = learning_params['max_epoch']
    trainer = learning_params['trainer']

    if model_name == 'mnist_infogan':
        output_dataset = MnistDataset()
        latent_spec = [
            (Uniform(62), False),
            (Categorical(10), True),
            (Uniform(1, fix_std=True), True),
            (Uniform(1, fix_std=True), True),
        ]
        model = MNISTInfoGAN(
            batch_size=batch_size,
            output_dataset=output_dataset,
            output_dist=MeanBernoulli(output_dataset.image_dim),
            latent_spec=latent_spec,
        )
    elif model_name == 'mnist_wasserstein':
        output_dataset = MnistDataset()
        latent_spec = [
            (Uniform(62), False),
        ]
        model = MNISTInfoGAN(
            batch_size=batch_size,
            output_dataset=output_dataset,
            output_dist=MeanBernoulli(output_dataset.image_dim),
            final_activation=None,
            latent_spec=latent_spec,
        )
    elif model_name == 'celebA_infogan':
        output_dataset = CelebADataset()
        latent_spec = [
            (Uniform(128), False),
            (Categorical(10), True),
            (Categorical(10), True),
            (Categorical(10), True),
            (Categorical(10), True),
            (Categorical(10), True),
            (Categorical(10), True),
            (Categorical(10), True),
            (Categorical(10), True),
            (Categorical(10), True),
            (Categorical(10), True),
        ]
        model = CelebAInfoGAN(
            batch_size=batch_size,
            output_dataset=output_dataset,
            output_dist=MeanGaussian(output_dataset.image_dim, fix_std=True),
            latent_spec=latent_spec,
        )
    elif model_name == 'celebA_wasserstein':
        output_dataset = CelebADataset()
        latent_spec = [
            (Uniform(128), False),
        ]
        model = CelebAInfoGAN(
            batch_size=batch_size,
            output_dataset=output_dataset,
            output_dist=MeanGaussian(output_dataset.image_dim, fix_std=True),
            final_activation=None,
            latent_spec=latent_spec,
        )
    elif model_name == 'horse_zebra':
        horse_dataset = HorseOrZebraDataset('horse')
        zebra_dataset = HorseOrZebraDataset('zebra')
        horse2zebra_model = Horse2Zebra_CycleGAN(
            input_dataset=horse_dataset,
            batch_size=batch_size,
            output_dataset=zebra_dataset,
            output_dist=MeanGaussian(zebra_dataset.image_dim, fix_std=True),
            final_activation=None,
            scope_suffix='_horse2zebra',
        )
        zebra2horse_model = Horse2Zebra_CycleGAN(
            input_dataset=zebra_dataset,
            batch_size=batch_size,
            output_dataset=horse_dataset,
            output_dist=MeanGaussian(zebra_dataset.image_dim, fix_std=True),
            final_activation=None,
            scope_suffix='_zebra2horse',
        )
    else:
        raise ValueError('Invalid model_name: {}'.format(model_name))

    if trainer == 'infogan':
        d_optim = tf.train.AdamOptimizer(2e-4, beta1=0.5)
        g_optim = tf.train.AdamOptimizer(1e-3, beta1=0.5)
        loss = GANLoss()
        loss_builder = InfoGANLossBuilder(
            model=model,
            loss=loss,
            batch_size=batch_size,
            g_optimizer=g_optim,
            d_optimizer=d_optim,

        )
        algo = trainer2.GANTrainer(
            loss_builder=loss_builder,
            exp_name=experiment_name,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            max_epoch=max_epoch,
            updates_per_epoch=updates_per_epoch,
        )

    elif trainer == 'wasserstein':
        d_optim = tf.train.AdamOptimizer(2e-4, beta1=0.5)
        g_optim = tf.train.AdamOptimizer(1e-3, beta1=0.5)
        loss = WassersteinGANLoss()
        loss_builder = GANLossBuilder(
            model=model,
            loss=loss,
            batch_size=batch_size,
            g_optimizer=g_optim,
            d_optimizer=d_optim,

        )
        algo = trainer2.WassersteinGANTrainer(
            loss_builder=loss_builder,
            exp_name=experiment_name,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            max_epoch=max_epoch,
            updates_per_epoch=updates_per_epoch,
        )

    elif trainer == 'test':
        d_optim = tf.train.AdamOptimizer(2e-4, beta1=0.5)
        g_optim = tf.train.AdamOptimizer(1e-3, beta1=0.5)
        loss = GANLoss()
        loss_builder = InfoGANLossBuilder(
            model=model,
            loss=loss,
            dataset=output_dataset,
            batch_size=batch_size,
            discrim_optimizer=d_optim,
            generator_optimizer=g_optim,
        )
        algo = trainer2.GANTrainer(
            loss_builder=loss_builder,
            exp_name=experiment_name,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            max_epoch=max_epoch,
            updates_per_epoch=updates_per_epoch,
        )
    elif trainer == 'cycle_gan':
        d_optim = tf.train.AdamOptimizer(2e-4, beta1=0.5)
        g_optim = tf.train.AdamOptimizer(2e-4, beta1=0.5)
        loss = LeastSquaresGANLoss()
        horse2zebra_loss_builder = GANLossBuilder(
            model=horse2zebra_model,
            loss=loss,
            batch_size=batch_size,
        )
        zebra2horse_loss_builder = GANLossBuilder(
            model=zebra2horse_model,
            loss=loss,
            batch_size=batch_size,
        )
        loss_builders = [horse2zebra_loss_builder, zebra2horse_loss_builder]
        loss_builder = CycleGANLossBuilder(
            loss_builders,
            g_optimizer=g_optim,
            d_optimizer=d_optim,
        )
        algo = trainer2.GANTrainer(
            loss_builder=loss_builder,
            exp_name=experiment_name,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            max_epoch=max_epoch,
            updates_per_epoch=updates_per_epoch,
        )
    else:
        raise ValueError('Invalid trainer: {}'.format(trainer))

    algo.train()
