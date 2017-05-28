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
@click.option('--n-samples', '-n',
              default=1,
              prompt='Please specify the number of samples',
              help='The number of samples.')
def sample(checkpoint_path, sampling_type, n_samples):
    import sample
    models_path = os.path.dirname(checkpoint_path)
    models_path = os.path.join(models_path, 'models.pkl')
    with open(models_path, 'rb') as f:
        models = pkl.load(f)
    for model in models:
        sample.sample(checkpoint_path, model, sampling_type, n_samples)

all_scripts.add_command(train)
