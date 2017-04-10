from __future__ import absolute_import
from __future__ import print_function
import tensorflow as tf
import os

from infogan.algos import trainer2
from infogan.algos.infogan_trainer import InfoGANTrainer, WassersteinGANTrainer
from infogan.algos.loss import GANLoss
from infogan.algos.loss_builder2 import InfoGANLossBuilder
from infogan.misc.datasets import MnistDataset, CelebADataset
from infogan.misc.distributions import Uniform, Categorical, MeanBernoulli, \
    MeanGaussian
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
            dataset=output_dataset,
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
            dataset=output_dataset,
            output_dist=MeanGaussian(output_dataset.image_dim, fix_std=True),
            final_activation=None,
            latent_spec=latent_spec,
        )
    else:
        raise ValueError('Invalid model_name: {}'.format(model_name))

    if trainer == 'infogan':
        algo = InfoGANTrainer(
            model=model,
            dataset=output_dataset,
            batch_size=batch_size,
            exp_name=experiment_name,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            max_epoch=max_epoch,
            updates_per_epoch=updates_per_epoch,
            info_reg_coeff=1.0,
            discrim_learning_rate=2e-4,
            generator_learning_rate=1e-3,
        )
    elif trainer == 'wasserstein':
        algo = WassersteinGANTrainer(
            model=model,
            dataset=output_dataset,
            batch_size=batch_size,
            exp_name=experiment_name,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
            max_epoch=max_epoch,
            updates_per_epoch=updates_per_epoch,
            info_reg_coeff=1.0,
            discrim_learning_rate=5e-5,
            generator_learning_rate=5e-5,
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
    else:
        raise ValueError('Invalid trainer: {}'.format(trainer))

    algo.train()
