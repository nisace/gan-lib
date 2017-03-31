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


#TODO: get sampling_type choices dynamically from Model object
@all_scripts.command(context_settings=dict(max_content_width=100))
@click.option('--checkpoint-path', '-p',
              prompt='Please specify the checkpoint file to load',
              help='The path of the checkpoint file to load (.ckpt).')
@click.option('--sampling-type', '-s',
              type=click.Choice(['random',
                                 'latent_code_influence',
                                 'linear_interpolation']),
              prompt='Please specify the sampling type',
              default='random',
              help='The type of sampling to perform:\n'
                   '- random: all samples are independent.\n'
                   '- latent_code_influence: the noise generator input is '
                   'the same along each column, except for one of the '
                   'latent distributions, for which the value varies along '
                   'the column.')
def sample(checkpoint_path, sampling_type):
    import sample
    model_path = os.path.dirname(checkpoint_path)
    model_path = os.path.join(model_path, 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pkl.load(f)
    sample.sample(checkpoint_path, model, sampling_type)


all_scripts.add_command(train)
