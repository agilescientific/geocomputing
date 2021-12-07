# Useful Slack channel info drops

## Python things

### Python cheatsheet(s)

- https://s3.amazonaws.com/assets.datacamp.com/blog_assets/PythonForDataScience.pdf - Good for the basics of python, and also covers some Numpy fundamentals.
- https://gto76.github.io/python-cheatsheet/ - Very comprehensive, but terse. Covers many things that we are not showing, some of which will need a dive into the documentation to understand.


### Collection summary

Summary of the different collections that we have looked at, which might be helpful:

* `list` - ordered, mutable, heterogeneous collection `[0, 1, 2, 3]`
* `tuple` - ordered, immutable, heterogeneous collection `(0, 1, 2, 3)`
* `str` (string) - ordered, immutable collection of characters `"This is a string"`
* `dict` (dictonary) - unordered mapping of immutable key to value of any type `{key1: value1, key3: value3, key2: value2}`
* `numpy.ndarray` (n-dimensional array) - 'lists with superpowers', can be multidimensional (convert from list: `np.array([0, 1, 2, 3])`)
* `set` - unordered collection of unique elements `{0, 4, 2, 3, 5}`. Use for set theory mathematics.


### Bracket summary

Different brackets:

`()` - immediately after a function/method: `func()`, sometimes with arguments: `np.max(arr1)`
     - Indicate `tuples`: `(1, 2, 3, 4)`
     - Order of operation: `(3 + 4) * 2`
     - Group logic statements: `if (a > b) and (b > c):`

`[]` - List creation: `[1, 2, 3, 4]`
     - Indexing into collections: `mylist[0]`, `mytuple[2]`, `myarr[:, 120]`
     - Slicing into collections: `myarr[::10, 30:50:5]`
     - Keying into a dictionary: `mydict[key]`
     - `list` comprehension: `[n**2 for n in range(1,11)]`

`{}` - Dictionary creation: `{key1: value1, key3: value3, key2: value2,}`
     - Set: `{0, 2, 1, 3, 8}`
     - f-strings: `f'The value of pi is: {math.pi:.4f}'`
     - `dict` comprehension: `{k: v * 100 for k, v in dict_a.items()}`


### Getting help

- In notebooks: `shift`+`tab` will give the documentation. Multiple presses will make it more persistant.
- In notebooks and iPython: `thing?` in a cell will print the help about `thing`.
- Anywhere with python: `help(thing)` will give you the documentation.


### Importing modules and libraries
There are three major ways to import packages:

* `import package` -> imports everything. Access things using `package.function()`.
* `import package as pkg` -> imports everything with an alias. Accessing things using `pkg.function()`.
* `from package import function1, function2` -> import only `function1` and `function2`. Accessed as `function1()` and `function2()`.


### Environment creation

To recap, here the environment creation process:

- Open an Anaconda prompt.
- Think of a name, I'm going to use `myenv`.
- Decide on a version of Python to use (if in doubt, go with 3.8 for now).
- Think of what you want in there. I'm going to create a fairly minimal environment containing `jupyter`, `numpy`, `scipy`, and `matplotlib`. I'm also adding `ipykernel` so we can use the magic 'add kernel' trick later.
- Type `conda create -n myenv python=3.8 jupyter numpy scipy matplotlib ipykernel` (To create an env containing "everything in Anaconda" -- which somone noticed does create a rather large environment -- you would use: `conda create -n myenv python-3.8 anaconda` but I generally don't do this.)
- Give it the feedback it wants and wait.
- Type conda activate myenv to get into the environment.
- Type `python -m ipykernel install --user --name myenv` (this is the only time you'll ever need to do this for this env)
- You're done!

You can also use the following, if you have an `environment.yml` file:
- `conda env create -f environment.yml` will create an environment with the packages in the environment.yml file. The name of the environment is specified in that file.
- Type `conda activate myenv` to get into the environment.
- Type `python -m ipykernel install --user --name myenv` (this is the only time you'll ever need to do this for this env)
- You're done!

## More tutorials!

- Agile’s X Lines of Python — https://github.com/agile-geoscience/xlines/tree/master/notebooks
- The SEG Geophysical Tutorials, almost all of them have Jupyter Notebooks with them, many of which are appropriate for beginners — https://wiki.seg.org/wiki/Geophysical_tutorials
- Here’s a playlist of tutorials from this year's Transform conference, all of them have notebooks etc. — https://www.youtube.com/playlist?list=PLgLft9vxdduCESA3xAo5Ts_ziO8vZAFG1
- Some tutorials from Jesse Pisel at UT Austin — https://github.com/jessepisel/5minutesofpython
- Some ‘kata’ (exercises) — as you complete each one, it gives you a new one to try — https://gist.github.com/kwinkunks/50f11dac6ab7ff8c3e6c7b34536501a2
- For something a bit different, here’s a nice collection of geothermal notebooks from Jan Niederau, as researcher at Aachen — https://github.com/Japhiolite/geothermics

## Jupyter Notebook things

### Markdown

The original Markdown page by John Gruber:
https://daringfireball.net/projects/markdown/
Getting started with markdown:
https://www.markdownguide.org/getting-started/
Cheat sheet:
https://www.markdownguide.org/cheat-sheet

Mathematical expressions in notebooks use LaTeX:
https://www.overleaf.com/learn/latex/Mathematical_expressions


### Restarting

A quick recap to get back into your jupyter notebook, if you didn't close anything down, you're good to go :heavy_check_mark:
To start fresh:

0: Open an Anaconda prompt (Windows) or a Terminal (Mac/Linux).
1: Run `conda activate geocomp`
2: Navigate to the `geocomp` folder by typing `cd /path/to/folder/geocomp/` (with the actual path! NB use backslashes on Windows). On Windows you can copy the path from your file explorer.
3: Type `jupyter notebook` to launch the Jupyter server.
4: In the browser window, click on the `notebooks` folder and then the notebook you need.
5: Re-run any cells containing things you need: `import`s, variable assignments, etc. Voilà, you're ready to code!


### Trust in Jupyter notebooks

Jupyter is running a webserver on your machine (your machine is called localhost and Jupyter is serving on port 8888 by default). Webservers need lots of security so people can't get in to places they shouldn't so Jupyter makes a random key for each session. Each notebook must send this key with every interaction with the server, so the server knows it's a legit notebook that it's allowed to talk to. This is how it knows which notebooks to trust (ones that it's running, basically).

So this is trustworthiness is not really a property of the notebook or who made it. It's more to do with the server wanting proof that it's allowed to listen to the messages the notebook is sending. It's akin to the cookie that a website puts on your computer to keep you logged in.


## Matplotlib

### Cheatsheets

The `matplotlib` cheatsheets are awesome! https://github.com/matplotlib/cheatsheets

### Colourmap bit:

Perceptually uniform sequential colormaps in `matplotlib`: `['viridis', 'plasma', 'inferno', 'magma', 'cividis']`. The roughly monchrome ramps are mostly fine too: `['gray', 'bone', 'pink', 'Blues']` and so on.

If you're not sure what I'm talking about, check this out: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html#sequential 

And if you're wondering why, I recommend reading this: https://agilescientific.com/blog/2017/12/14/no-more-rainbows?rq=colormaps or https://www.kennethmoreland.com/color-advice/ 

If you prefer watching than reading, which is totally fine as we're talking colourmaps ,-) then watch this old scipy 2015 video: https://www.youtube.com/watch?v=xAoljeRJ3lU

If you insist on jet, then at least let's have turbo: https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html

And also check out Petrosys' view on this: https://www.petrosys.com.au/using-colour-to-communicate/


## Pandas things

### `df.loc` vs `df.iloc`

`df.loc` is accessing the datafame by the *index names* (row indices and column indices), while `df.iloc` is accessing by *position*, similar to a `list` or `np.adarray`). More details are here: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#different-choices-for-indexing


## Other things

### Open source licenses

Open source licenses can be confusing. Here is a good overview of some of the most common: https://opensource.org/licenses

### GIS
#### Raster data:
`GDAL` (written in C but has bindings for python) - https://pcjericks.github.io/py-gdalogr-cookbook/ is a good starting point.
`Rasterio` (built on top of GDAL, a bit easier to use) - https://rasterio.readthedocs.io/en/latest/intro.html
For processing the raster images: `scikit-image`, `scipy.ndimage` - https://scikit-image.org/ https://docs.scipy.org/doc/scipy/reference/ndimage.html
#### Vector data:
`geopandas` (an extension of `pandas` that loads shapefiles and similar vector data) - https://geopandas.org/
`GDAL`/`OGR` can handle vector data as well, but geopandas is a bit nicer.
#### Utility packages:
`cartopy` (for handling CRS transforms and similar projection things, built on (non-python) `proj`) - https://scitools.org.uk/cartopy/docs/latest/
`folium` (making interactive `leaflet.js` maps in python) - https://python-visualization.github.io/folium/

