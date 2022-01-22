# geocomputing

This is the main repository for Agile's geocomputing courses.

The main features:

- There is one main script, `geocomputing.py`, which execute one of three commands:
  - `build` &mdash; Build a course or courses. This command has several options, see `./geocomputing.py build --help`.
  - `clean` &mdash; Remove the build files associated with a course or courses.
  - `publish` &mdash; Publish a course or courses to AWS.
  - `test` &mdash; Test that a course builds.
- All of the commands take either a single course name, or the `--all` flag, which applies the command to all the courses listed in `all.yaml`.
- There is one control file per course, e.g. `geocomp.yaml`. This file contains the metadata for the course, including the curriculum and a list of its notebooks.
- There is one over-arching control file, `config.yaml`. This file contains a default group, `'production'`, which lists all the courses that will be built by the `publish` command with its default argument.
- There is one main, common environment file, `environment.yaml`. This contains packages to be installed for (i.e. common to)  all courses. A course's YAML control file lists any other packages to install for that class.


## Requirements

In order to build files, you will need the following:

- Python 3.9+.
- Everything in `dev_requirements.txt`.
- If you want to upload to AWS S3, you'll need to install `boto3` and `botocore` as well, and have credentials set up on your machine. The easiest way to manage an AWS environment on your computer is probably via the AWS CLI.


## Usage

To see high-level help:

    ./geocomputing.py


### Usage of `build`

Run the `geocomputing.py` script like this to build the `geocomp` (_Intro to Geocomputing_) class:

    ./geocomputing.py build geocomp

You can build any course for which a YAML file exists. So the command above will compile the course specified by `geocomp.yaml`.

All of the commands can take the option `--all`. This will apply the command to all of the courses listed in `all.yaml`. In this case, don't pass any individual course name.

In addition, you can pass the following options:

- **`--clean` / `--no-clean`** &mdash; Whether to delete the build files. Default: `clean`.
- **`--zip` / `--no-zip`** &mdash; Whether to **create** the zip file for the course repo. Default: `zip`.
- **`--upload` / `--no-upload`** &mdash; Whether to **upload** the zip file to `geocomp.s3.amazonaws.com`. Default: `no-upload`. Note that this requires AWS credentials to be set up on your machine.
- **`--clobber` / `--no-clobber`** &mdash; Whether to silently overwrite existing ZIP file and/or build directory. If `no-clobber`, the CLI will prompt you to overwrite or not. Default: `no-clobber`.

To build the machine learning course, silently overwriting any existing builds on your system:

    ./geocomputing.py build geocomp-ml --clobber


### Usage of `clean`

Cleans the build files for a course. I.e. everything in `build` and its ZIP file.

    ./geocomputing.py clean geocomp-ml


### Usage of `publish`

Publish a course, or those listed in `all.yaml`. The ZIP file(s) will be uploaded to AWS. For example, to publish all the courses:

    ./geocomputing.py publish --all


### Usage of `test`

Tests that a specific course builds, leaving no sawdust, or use the `--all` option to test all courses in `all.yaml`. This command builds a course, does not make a ZIP, does not uplad anything, and removes the build folder. (To keep the build folder or make a zip, use the `build` command with the appropriate options, see above.) Here's how to test the machine learning course:

    ./geocomputing.py test geocomp-ml

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


## TODO

- More testing (I've only really worked on `geocomp` and `geocomp-ml` in development).
- Documentation of the control file options.
- Proper tagging of `demo` notebooks, which is a new feature (testing in **Advanced_functions.ipynb**)
- A thorough review of the content in each course.
