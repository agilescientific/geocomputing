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
import warnings
from urllib.request import urlretrieve

import requests
import click
import yaml
from jinja2 import Environment, FileSystemLoader

try:
    import boto3
    from botocore.exceptions import ClientError
    aws_available = True
except:
    aws_available = False

from customize import process_notebook

env = Environment(loader=FileSystemLoader('templates'))


@click.command(context_settings=dict(help_option_names=['--help', '-h']))
@click.argument('course')
@click.option('--clean/--no-clean', default=True, help="Delete the build dir? Default: clean.")
@click.option('--zip/--no-zip', default=True, help="Make the zip file? Default: zip.")
@click.option('--upload/--no-upload', default=False, help="Upload the ZIP to S3? Default: no-upload.")
@click.option('--clobber/--no-clobber', default=False, help="Clobber existing files? Default: no-clobber.")
def build(course, clean, zip, upload, clobber):
    """
    Main building function. Compiles the required files into a course repo, which
    will be zipped by default.
    
    Args:
        course (str): The course to build. One of geocomp, geocomp-ml, geocomp-gp, etc.
        clean (bool): Whether to remove the build files. Default: True.
        zip (bool): Whether to create the zip file for the course repo. Default: True.
        clobber (bool): Whether to overwrite existing ZIP file and build directory. 
            If False, the CLI will prompt you to confirm overwrite. Default: False.
    Returns:
        None.
    """
    # Read the YAML control file.
    with open(f"{course.removesuffix('.yaml')}.yaml", 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(e)
    config['course'] = course

    # Make a path to store everything.
    build_path = 'build'
    path = pathlib.Path(build_path).joinpath(course)
    if path.exists():
        if clobber or click.confirm("The target directory exists and will be overwritten. Are you sure?", default=True, abort=True):
            shutil.rmtree(path)
    if pathlib.Path(f"{course}.zip").exists():
        if clobber or click.confirm("The ZIP file exists and will be overwritten. Are you sure?", default=True, abort=True):
            pathlib.Path(f"{course}.zip").unlink()

    _ = path.mkdir(parents=True, exist_ok=True)

    # Build the notebooks; also deals with images.
    *paths, _, data_urls_to_check = build_notebooks(path, config)

    # Check the data files exist.
    click.echo('Checking and downloading data ', nl=False)
    for url in data_urls_to_check:
        click.echo('.', nl=False)
        if requests.head(url).status_code != 200:
            raise Exception(f"Missing data URL: {url}")

    # Make the data directory.
    build_data(path, config)
    click.echo()

    # Deal with scripts.
    if scripts := config.get('scripts'):
        for script in scripts:
            for p in paths:
                shutil.copyfile(pathlib.Path('scripts') / script, p / script)

    # Make the references folder.
    if refs := config.get('references'):
        ref_path = path.joinpath('references')
        ref_path.mkdir()
        for fname in refs:
            shutil.copyfile(pathlib.Path('references') / fname, ref_path / fname)

     # Make the environment.yaml file.
    build_environment(path, config)

    # Make the README.
    build_readme(path, config)

    # Zip it.
    if zip:
        zipped = shutil.make_archive(course, 'zip', root_dir=build_path, base_dir=course)
        click.echo(f"Created {zipped}")

    # Upload to AWS.
    if upload:
        success = upload_zip(zipped)
        if success:
            click.echo(f"Uploaded {zipped}")

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
    # Make the various directories.
    m_path = path.joinpath('master')
    m_path.mkdir()
    nb_path = path.joinpath('notebooks')
    nb_path.mkdir()
    if config.get('demos'):
        demo_path = path.joinpath('demos')
        demo_path.mkdir()
    else:
        demo_path = None

    all_items = [f for items in config['curriculum'].values() for f in items]
    notebooks = list(filter(lambda item: '.ipynb' in item, all_items))
    notebooks += config.get('extras', list())
    images_to_copy = []
    data_urls_to_check = []
    click.echo('Processing notebooks ', nl=False)
    for notebook in notebooks:
        infile = pathlib.Path('prod') / notebook
        outfile = nb_path / notebook
        images, data_urls = process_notebook(infile, outfile)
        images_to_copy.extend(images)
        data_urls_to_check.extend(data_urls)
        shutil.copyfile(infile, m_path / notebook)
        # Clear the outputs in the master file.
        _ = os.system("nbstripout {}".format(m_path / notebook))
        click.echo('+', nl=False)
    notebooks = config.get('demos', list())
    for notebook in notebooks:
        infile = pathlib.Path('prod') / notebook
        outfile = demo_path / notebook
        images, data_urls = process_notebook(infile, outfile, demo=True)
        images_to_copy.extend(images)
        data_urls_to_check.extend(data_urls)
        shutil.copyfile(infile, m_path / notebook)
        # Clear the outputs in the master file.
        _ = os.system("nbstripout {}".format(m_path / notebook))
        click.echo('+', nl=False)
    click.echo()
    if images_to_copy:
        img_path = path.joinpath('images')
        img_path.mkdir()
        for image in images_to_copy:
            shutil.copyfile(pathlib.Path('images') / image, img_path / image)

    return m_path, nb_path, demo_path, data_urls_to_check


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
    return

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
    return


def build_data(path, config):
    """Build the data directory. Files must exist in the data folder
    of the geocomp bucket of AWS S3.
    """
    data_path = path.joinpath('data')
    data_path.mkdir()

    data_url = config.get('data_url', "https://geocomp.s3.amazonaws.com/data/")

    if datasets := config.get('data'):
        for fname in datasets:
            click.echo('+', nl=False)
            fpath = data_path / fname
            if not fpath.exists():
                url = f"{data_url}{fname}"
                urlretrieve(url, fpath)
            if fpath.suffix == '.zip':
                # Inflate and delete the zip.
                with zipfile.ZipFile(fpath, 'r') as z:
                    z.extractall(data_path)
                fpath.unlink()
    else:
        data_path.joinpath('folder_should_be_empty.txt').touch()
    return


def upload_zip(file_name, bucket='geocomp', object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        args = {'ACL':'public-read'}
        _ = s3_client.upload_file(file_name, bucket, object_name, ExtraArgs=args)
    except ClientError as e:
        warnings.warn("Upload to S3 failed:", e)
        return False
    return True


if __name__ == '__main__':
    build()
