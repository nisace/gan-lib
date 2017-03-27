import os

import cPickle as pkl
import click
import yaml


@click.group()
def all_scripts():
    """ This runs the different functionality of gan-lib. """
    pass


@all_scripts.command()
@click.option('--params-file', '-p',
              type=click.File('r'),
              prompt='Please specify a YAML parameters file',
              help='The path of the YAML parameters file.')
def train(params_file):
    import run
    params = yaml.load(params_file.read())
    params_file.close()
    run.train(**params)


@all_scripts.command()
@click.option('--checkpoint-path', '-p',
              prompt='Please specify the checkpoint file to load',
              help='The path of the checkpoint file to load (.ckpt).')
@click.option('--visualize-latent-code',
              is_flag=True,
              # prompt='Please specify if you want to visualize the latent '
              #        'codes influence',
              help='If True, sample with varying latent codes to see their '
                   'influence.')
def sample(checkpoint_path, visualize_latent_code):
    import sample
    model_path = os.path.relpath(checkpoint_path, 'ckt')
    model_path = os.path.dirname(os.path.join('logs', model_path))
    model_path = os.path.join(model_path, 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pkl.load(f)
    z_value = None
    if visualize_latent_code:
        z_value = model.get_z_value()
    sample.sample(checkpoint_path, model.image_shape, model, z_value=z_value)


all_scripts.add_command(train)
