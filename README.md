# geocomputing

This is the main repository for Agile's geocomputing courses.

The main features:

- There is one control file per course, e.g. `geocomp.yaml`. This file contains the metadata for the course, including the curriculum and a list of its notebooks.
- There is one main environment file, `environment.yaml`. This contains packages to be installed for all classes. (A class's control file lists any other packages to install for that class.)
- There is one main script, `build.py`, which builds a course's ZIP file. Run it from the command line, with the class to build.


## Usage

Run the `build.py` script like this to build the `geocomp` (_Intro to Geocomputing_) class:

    ./build.py geocomp

### Options

- **`--clean`** &mdash; Whether to remove the build files. Default: `True`.
- **`--zip`** &mdash; Whether to create the zip file for the course repo. Default: `True`.
- **`--clobber`** &mdash; Whether to overwrite existing ZIP file and build directory. If False, the CLI will prompt you to confirm overwrite. Default: `False`.


## Example control file

A course must have a YAML control file containing something like the following example of a 2-day course:

```yaml
title: Introduction to Python for Geologists
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

- Check the actual output :D
- Thorough testing (I've only really worked on `geocomp` and `geocomp-ml` in development).
- Documentation of the control file options.
- Proper tagging of `demo` notebooks, which is a new feature (testing in **Advanced_functions.ipynb**)
- A thorough review of the content in each course.
