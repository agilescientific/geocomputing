# geocomputing

[![test-builds](https://github.com/agilescientific/geocomputing/actions/workflows/test-all.yml/badge.svg)](https://github.com/agilescientific/geocomputing/actions/workflows/test-all.yml)
[![publish-classes-to-S3](https://github.com/agilescientific/geocomputing/actions/workflows/publish.yml/badge.svg)](https://github.com/agilescientific/geocomputing/actions/workflows/publish.yml)

This is the main repository for Agile's geocomputing courses.


## Requirements

In order to build files, you will need the following:

- Python 3.9+.
- The course-building package, `kosu`. To install it:

```shell
pip install kosu
```

## Usage

To see high-level help:

    kosu --help


### Usage of `build`

Run `kosu` on the command line to build the `geocomp` (_Intro to Geocomputing_) class:

    kosu build geocomp

You can build any course for which a YAML file exists. So the command above will compile the course specified by `geocomp.yaml`.

All of the commands can take the option `--all`. This will apply the command to all of the courses listed under `all` in `.kosu.yaml`. In this case, don't pass any individual course name.

In addition, you can pass the following options:

- **`--clean` / `--no-clean`** &mdash; Whether to delete the build files. Default: `clean`.
- **`--zip` / `--no-zip`** &mdash; Whether to **create** the zip file for the course repo. Default: `zip`.
- **`--upload` / `--no-upload`** &mdash; Whether to **upload** the zip file to `geocomp.s3.amazonaws.com`. Default: `no-upload`. Note that this requires AWS credentials to be set up on your machine.
- **`--clobber` / `--no-clobber`** &mdash; Whether to silently overwrite existing ZIP file and/or build directory. If `no-clobber`, the CLI will prompt you to overwrite or not. Default: `no-clobber`.

To build the machine learning course, silently overwriting any existing builds on your system:

    kosu build geocomp-ml --clobber


### Usage of `clean`

Cleans the build files for a course. I.e. everything in `build` and its ZIP file.

    kosu clean geocomp-ml


### Usage of `publish`

Publish a course, or those listed in `all.yaml`. The ZIP file(s) will be uploaded to AWS. For example, to publish all the courses:

    kosu publish --all


### Usage of `test`

Tests that a specific course builds, leaving no sawdust, or use the `--all` option to test all courses in `all.yaml`. This command builds a course, does not make a ZIP, does not uplad anything, and removes the build folder. (To keep the build folder or make a zip, use the `build` command with the appropriate options, see above.) Here's how to test the machine learning course:

    kosu test geocomp-ml

There is an option `--environment` that will also generate an environment file called `environment-all.yml`. (This is used for automated testing on GitHub.)

In general, if a course does not build, the script will throw an error. It does not try to deal with or interpret the error or explain what's wrong.


## Example control file

A course must have a YAML control file containing something like the following example of a 2-day course:

```yaml
title: Introduction to Python for Geologists
environment: geogeol  # Only if different from course name.
conda:  # Extra conda packages, as well as all of standard geocomp env.
  - verde
pip:  # Extra pip packages.
  - striplog
data:
  - sussex.zip  # Will be unzipped.
  - B-41_tops.txt
data_url: https://geocomp.s3.amazonaws.com/data/  # This is the default value.
scripts:
  - utils.py  # Added to `master` and `notebooks` folders (not `demos`).
curriculum:
  1:  # Day 1.
    - Course overview
    - The Python interpreter and the IPython environment
    - Jupyter Notebooks
    - Intro_to_Python.ipynb
    - Check out and feedback
  2:  # Day 2.
    - Check in and review
    - Intro_to_Python.ipynb  # .ipynb files will be added to `notebooks`.
    - Check out and feedback
extras:  # These will be added to `notebooks` and listed in the Curriculum.
  - Intro_to_NumPy.ipynb
  - Seismic_data_basics.ipynb
  - Pandas_for_data_management.ipynb
  - Read_and_write_LAS.ipynb
demos:  # These will be added to `demos` and NOT listed in the Curriculum.
  - Birthquake.ipynb
  - Volumetrics_and_units.ipynb
```

Only `title` and `curriculum` are required fields.
