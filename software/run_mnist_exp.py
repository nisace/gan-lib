from __future__ import absolute_import
from __future__ import print_function

import os

import datetime
import dateutil.tz

from infogan.algos.infogan_trainer import InfoGANTrainer
from infogan.misc.datasets import Cifar10Dataset, MnistDataset
from infogan.misc.distributions import Uniform, Categorical, MeanBernoulli, \
    Gaussian
from infogan.models.regularized_gan import MNISTRegularizedGAN, \
    CIFAR10RegularizedGAN, CelebAInfoGANRegularizedGAN, LSUN_DCGAN

if __name__ == "__main__":

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    root_log_dir = "logs/mnist"
    root_checkpoint_dir = "ckt/mnist"
    batch_size = 128
    updates_per_epoch = 100
    max_epoch = 500

    exp_name = "mnist_%s" % timestamp

    log_dir = os.path.join(root_log_dir, exp_name)
    checkpoint_dir = os.path.join(root_checkpoint_dir, exp_name)

    os.makedirs(log_dir)
    os.makedirs(checkpoint_dir)

    # dataset = MnistDataset()
    # latent_spec = [
    #     (Uniform(62), False),
    #     (Categorical(10), True),
    #     (Uniform(1, fix_std=True), True),
    #     (Uniform(1, fix_std=True), True),
    # ]
    # model = MNISTRegularizedGAN(
    #     output_dist=MeanBernoulli(dataset.image_dim),
    #     latent_spec=latent_spec,
    #     batch_size=batch_size,
    #     image_shape=dataset.image_shape,
    # )

    dataset = Cifar10Dataset()
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
        # (Uniform(1, fix_std=True), True),
        # (Uniform(1, fix_std=True), True),
        # (Uniform(1, fix_std=True), True),
        # (Uniform(1, fix_std=True), True),
    ]
    model = CelebAInfoGANRegularizedGAN(
        output_dist=Gaussian(dataset.image_dim, fix_std=True),
        latent_spec=latent_spec,
        batch_size=batch_size,
        image_shape=dataset.image_shape,
    )
    # model = CIFAR10RegularizedGAN(
    #     output_dist=MeanBernoulli(dataset.image_dim),
    #     latent_spec=latent_spec,
    #     batch_size=batch_size,
    #     image_shape=dataset.image_shape,
    # )

    algo = InfoGANTrainer(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        exp_name=exp_name,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        max_epoch=max_epoch,
        updates_per_epoch=updates_per_epoch,
        info_reg_coeff=1.0,
        generator_learning_rate=1e-3,
        discriminator_learning_rate=2e-4,
    )

    algo.train()
