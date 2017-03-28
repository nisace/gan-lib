# GAN-lib

This repository was originally copied from https://github.com/openai/InfoGAN, the official implementation of https://arxiv.org/abs/1606.03657.
The current version adds the following:
- CelebA experiment (the original repository contained MNIST experiment only)
- Wasserstein GAN (https://arxiv.org/abs/1701.07875)
- Sampling from a trained model

The next steps are to add experiments (ImageNet, CIFAR) and implement GAN variants like CatGAN or Unrolled GAN. 

## Running in Docker

To run in docker, use the docker.sh script. You can build the image or 
run the container and go inside it. Both commands are available in CPU and GPU versions. 

```bash
$ git clone git@github.com:nisace/gan-lib.git
$ cd gan-lib/
$ ./docker.sh {build, run} {cpu, gpu}
root@X:/gan-lib#
```

## Command line interface

All scripts are accessible through manage.py. The see help, execute the following command:

```bash
python manage.py --help
```

## Training a model

```bash
python manage.py train -p {params/mnist.yml, params/celebA.yml, params/mnist_wasserstein_.yml, params/celebA_wasserstein.yml,...}
```

This will train the model and save it into ./ckpt/experiment_name/experiment_name_with_date/

Logs are saved in ./logs/experiment_name/experiment_name_with_date/

## Seeing results

You can launch TensorBoard to view the generated images:

```bash
tensorboard --logdir logs/
```

or give a specific logdir:

```bash
tensorboard --logdir logs/mnist_infogan/mnist_infogan_2017_03_20_10_49_54/
```

## Sampling from a trained model

You can sample from a trained model:

```bash
python manage.py sample -p path/to/checkpoint.ckpt -s {random, latent_code_influence}
```

Example:

```bash
python manage.py sample -p ckt/mnist_infogan/mnist_infogan_2017_03_20_10_49_54/mnist_infogan_2017_03_20_10_49_54_400.ckpt -s random
```

## Examples

### MNIST:

![result](software/samples/mnist.png)

### CelebA:

![result](software/samples/celebA.png)

### MNIST Wassertein:

![result](software/samples/mnist_wasserstein.png)

![result](software/samples/mnist_wasserstein_generator_loss.png)

### CelebA Wassertein:

![result](software/samples/celebA_wasserstein.png)

![result](software/samples/celebA_wasserstein_generator_loss.png)
