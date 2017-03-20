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
def sample(checkpoint_path):
    import sample
    sample.sample(checkpoint_path)


all_scripts.add_command(train)
