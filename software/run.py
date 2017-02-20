from __future__ import absolute_import
from __future__ import print_function

import os

import datetime
import dateutil.tz

from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.datasets import MnistDataset, CelebADataset
from infogan.misc.distributions import Uniform, Categorical, MeanBernoulli, \
    Gaussian
from infogan.models.regularized_gan import MNISTInfoGAN, \
    CelebAInfoGAN
from utils.file_system_utils import make_exists


def train(dataset_name):
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = os.path.join('logs', dataset_name)
    root_checkpoint_dir = os.path.join('ckt', dataset_name)
    experiment_name = '{}_{}'.format(dataset_name, timestamp)
    log_dir = os.path.join(root_log_dir, experiment_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, experiment_name)
    make_exists(log_dir)
    make_exists(checkpoint_dir)

    batch_size = 128
    updates_per_epoch = 100
    max_epoch = 500

    if dataset_name == 'mnist':
        dataset = MnistDataset()
        latent_spec = [
            (Uniform(62), False),
            (Categorical(10), True),
            (Uniform(1, fix_std=True), True),
            (Uniform(1, fix_std=True), True),
        ]
        model = MNISTInfoGAN(
            output_dist=MeanBernoulli(dataset.image_dim),
            latent_spec=latent_spec,
            batch_size=batch_size,
            image_shape=dataset.image_shape,
        )

    elif dataset_name == 'celebA':
        dataset = CelebADataset()
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
            output_dist=Gaussian(dataset.image_dim, fix_std=True),
            latent_spec=latent_spec,
            batch_size=batch_size,
            image_shape=dataset.image_shape,
        )
    else:
        raise ValueError('Invalid dataset_name: {}'.format(dataset_name))

    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        exp_name=experiment_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_epoch=max_epoch,
        updates_per_epoch=updates_per_epoch,
        info_reg_coeff=1.0,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=2e-4,
    )

    algo.train()
