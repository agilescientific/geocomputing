#!/usr/bin/env python
#-*- coding: utf-8 -*-
"""
Author: Agile Scientific
Licence: Apache 2.0

To use the CLI type this on the command line:
    build --help
"""
import pathlib
import shutil
import zipfile
import os
from urllib.request import urlretrieve

import click
import yaml
from jinja2 import Environment, FileSystemLoader

from customize import process_notebook

env = Environment(loader=FileSystemLoader('templates'))


@click.command(context_settings=dict(help_option_names=['--help', '-h']))
@click.argument('course')
@click.option('--clean', default=True, help="Whether to delete the build directory.")
@click.option('--zip', default=True, help="Whether to make the zip file.")
def build(course, clean, zip):
    """Main building function"""

    # Read the YAML control file.
    with open(f'{course}.yaml', 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    config['course'] = course

    # Make a path to store everything.
    path = pathlib.Path('build').joinpath(course)
    _ = path.mkdir(parents=True, exist_ok=True)

    # Build the notebooks.
    paths = build_notebooks(path, config)

    # Make the data directory.
    build_data(path, config)

    # Deal with scripts.
    if scripts := config.get('scripts'):
        for script in scripts:
            for p in paths:
                shutil.copyfile(pathlib.Path('scripts') / script, p / script)

    # Make the environment.yaml file.
    build_environment(path, config)

    # Make the README.
    build_readme(path, config)

    # Zip it.
    if zip:
        zipped = shutil.make_archive(course, 'zip', path)
        click.echo(f"Created {zipped}")

    # Remove build.
    if clean:
        shutil.rmtree(path)
        click.echo(f"Removed build files.")

    return

def build_notebooks(path, config):
    """
    Process the notebook files. We'll look at three sections of the
    config: curriculum (which contains non-notebook items too),
    extras (which are listed in the README), and demos (which are not).
    """
    # Make the master directory.
    m_path = path.joinpath('master')
    m_path.mkdir(exist_ok=True)

    # Make the various notebooks directories.
    nb_path = path.joinpath('notebooks')
    nb_path.mkdir(exist_ok=True)
    demo_path = path.joinpath('demos')
    demo_path.mkdir(exist_ok=True)

    all_items = [f for items in config['curriculum'].values() for f in items]
    notebooks = list(filter(lambda item: '.ipynb' in item, all_items))
    notebooks += config['extras']
    click.echo('Processing notebooks ', nl=False)
    for notebook in notebooks:
        infile = pathlib.Path('prod') / notebook
        outfile = nb_path / notebook
        process_notebook(infile, outfile)
        shutil.copyfile(infile, m_path / notebook)
        # Clear the outputs in the master file.
        _ = os.system("nbstripout {}".format(m_path / notebook))
        click.echo('+', nl=False)
    notebooks = config.get('demos', list())
    for notebook in notebooks:
        infile = pathlib.Path('prod') / notebook
        outfile = demo_path / notebook
        process_notebook(infile, outfile, demo=True)
        shutil.copyfile(infile, m_path / notebook)
        # Clear the outputs in the master file.
        _ = os.system("nbstripout {}".format(m_path / notebook))
        click.echo('+', nl=False)
    click.echo()

    return m_path, nb_path, demo_path


def build_environment(path, config):
    """Construct the environment.yaml file for this course."""
    # Get the base environment.
    with open(f'environment.yaml', 'r') as f:
        try:
            deps = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)

    # Now add the course-specific stuff from the config.
    conda = {'name': config['course']}  # This puts name first.
    conda.update(deps)
    if isinstance(conda['dependencies'][-1], dict):
        pip = conda['dependencies'].pop()
    if p := config.get('pip'):
        pip['pip'].extend(p)
    if c := config.get('conda'):
        conda['dependencies'].extend(c)
    conda['dependencies'].append(pip)

    # Write the new environment file to the course directory.
    with open(path / 'environment.yaml', 'w') as f:
        f.write(yaml.dump(conda, default_flow_style=False, sort_keys=False))


def build_readme(path, config):
    """Build the README.md using Jinja2 templates. Note that there
    is a reasonable amount of magic happening at the template level,
    especially text formatting. So if you're looking to change
    something in the README.md, it's probably in there somewhere.
    """
    content = dict(env=config['course'],
                   title=config['title'],
                   curriculum=config.get('curriculum'),
                   extras=config.get('extras'),
                  )
    template = env.get_template('README.md')
    with open(path / 'README.md', 'w') as f:
        f.write(template.render(**content))


def build_data(path, config):
    """Build the data directory. Files must exist in the data folder
    of the geocomp bucket of AWS S3.
    """
    data_path = path.joinpath('data')
    data_path.mkdir(exist_ok=True)

    if datasets := config.get('data'):
        click.echo('Downloading data ', nl=False)
        for fname in datasets:
            click.echo('+', nl=False)
            fpath = path / fname
            if not fpath.exists():
                url = f"https://geocomp.s3.amazonaws.com/data/{fname}"
                urlretrieve(url, fpath)
            if fpath.suffix == '.zip':
                # Inflate and delete the zip.
                with zipfile.ZipFile(fpath, 'r') as z:
                    z.extractall(path)
                fpath.unlink()
    else:
        path.joinpath('folder_should_be_empty.txt').touch()
    click.echo()


if __name__ == '__main__':
    build()
