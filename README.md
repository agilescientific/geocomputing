# geocomputing

This is the main repository for Agile's geocomputing courses.

The main features:

- There is one control file per course, e.g. `geocomp.yaml`. This file contains the metadata for the course, including the curriculum and a list of its notebooks.
- There is one main environment file, `environment.yaml`. This contains packages to be installed for all classes. (A class's control file lists any other packages to install for that class.)
- There is one main script, `build.py`, which builds a course's ZIP file. Run it from the command line, with the class to build.


## Usage

Run the `build.py` script like this to build the `geocomp` (_Intro to Geocomputing_) class:

    ./build.py geocomp


## TODO

- Images.
- Thorough testing (I've only really worked on `geocomp` and `geocomp-ml` in development).
- Documentation of the control file options.
- Proper tagging of `demo` notebooks, which is a new feature.
- A thorough review of the content in each course.
