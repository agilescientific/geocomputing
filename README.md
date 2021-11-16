% welcome

## Set up an environment

To build an environment called `geoml`, type the following commands inside the Anaconda Prompt (or terminal) **in the same directory where you placed the repo** (i.e. so that the `environment.yml` file is in the working directory):

```
conda env create
conda activate geoml
python -m ipykernel install --user --name geoml
```

Now you can start a notebook server with `jupyter notebook` and you're all set!
