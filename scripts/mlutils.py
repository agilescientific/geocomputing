#!/usr/bin/env python
"""
Various utilities for Agile's machine learning classes.
"""
import os
import glob
import random
import itertools
from collections import namedtuple

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import matplotlib.ticker as plticker
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
import seaborn as sns
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

def gen_knn_data(size=(20, 20)):
    """
    Make data for KNN plotter.
    """
    a, b = size
    X_0 = np.random.multivariate_normal([10,10], [[3, 0],[0, 3]], size=a)
    X_1 = np.random.multivariate_normal([13,7], [[3, 0],[0, 3]], size=b)
    X = np.vstack([X_0, X_1])
    y = np.array(a*[0] + b*[1])
    return X, y


def plot_knn(size=(20, 20), k=7, target=(14, 11)):
    """
    A figure to illustrate how the k-nearest neighbours estimator works.
    """
    X, y = gen_knn_data(size)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(*X.T, c=y, cmap='RdBu', vmin=-0.3, vmax=1.4, s=60)
    ax.axis('equal')
    ax.grid(c='k', alpha=0.2)
    ax.legend(*scatter.legend_elements())

    target = np.atleast_2d(target)

    dists, = cdist(target, X)
    idx = np.argsort(dists)[:k]
    r = dists[idx[-1]]
    nearest = X[idx]

    y_pred = mode(y[idx]).mode.item()

    ax.scatter(*target.T, c='k', s=120, marker='x')
    ax.scatter(*nearest.T, ec='k', s=130, fc='none')
    ax.set_title(f"Prediction: {y_pred}", fontsize=20)
    circle = plt.Circle(np.squeeze(target), radius=r, color='lightgray', zorder=0)
    ax.add_artist(circle)

    plt.show()
    return


def decision_regions(clf, X_val, y_val, extent, step=1):
    """
    Generate the decision surface of a classifier.
    Args:
        clf: sklearn classifier
        X_val (ndarray): The validation features, n_samples x n_features.
        y_val (ndarray): The validation labels, n_samples x 1.
        extent (tuple or list-like): floats (left, right, bottom, top). 
            The bounding box in data coordinates that the image will fill.
    Returns:
        y_pred (ndarray): The predicted labels using X_val, n_samples x 1
        y_all (ndarray): A 2D array representing whether each point lies above
            or below the decision boundary (hyperplane) as well as how far it is 
            from that boundary.
    """
    y_pred = clf.predict(X_val)
    x_min, x_max, y_min, y_max = extent
    try:
        x_step, y_step = step
    except TypeError:
        x_step = y_step = step
    xx, yy = np.meshgrid(np.arange(x_min, x_max, x_step),
                         np.arange(y_min, y_max, y_step))
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    y_all = Z.reshape(xx.shape + (-1,))
    return y_pred, y_all


def visualize(X_val, y_val, y_prob, cutoff=0.5, ncols=6, nrows=3, figsize=(12, 8), classes=None, shape=None):
    """
    Visualize some random samples from the prediction results.
    Colours: green for a good prediction, red for a wrong one. If the
    probability was less than some cutoff (default 0.5), we'll mute the colour.

    Args:
        X_val (ndarray): The validation features, n_samples x n_features.
        y_val (ndarray): The validation labels, n_samples x 1.
        y_prob (ndarray): The predicted probabilities, n_samples x n_classes.
        cutoff (float): the cutoff for 'uncertain'.
        ncols (int): how many plots across the grid.
        nrows (int): how many plots down the grid.
        figsize (tuple): tuple of ints.
        classes (array-like): the classes, in order. Will be inferred if None.
        shape (tuple): Shape of each instance, if it needs reshaping.
    """
    idx = random.sample(range(X_val.shape[0]), ncols*nrows)
    sample = X_val[idx]

    if classes is None:
        classes = np.unique(y_val)
    else:
        y_val = np.asarray(classes)[y_val]

    fig, axs = plt.subplots(figsize=figsize, ncols=ncols, nrows=nrows)
    axs = axs.ravel()

    for ax, img, actual, probs in zip(axs, sample, y_val[idx], y_prob[idx]):

        pred = classes[np.argmax(probs)]
        prob = np.max(probs)
        if shape is not None:
            img = img.reshape(shape)

        ax.imshow(np.squeeze(img), cmap='gray')
        ax.set_title(f"{pred} - {prob:.3f}\n[{actual}]")
        ax.set_xticks([])
        ax.set_yticks([])

        if prob > cutoff:
            c = 'limegreen' if (actual == pred) else 'red'
        else:
            c = 'y' if (actual == pred) else 'lightsalmon'

        for spine in ax.spines.values():
            spine.set_edgecolor(c)
            spine.set_linewidth(4)

    return


def get_file_info(fname):
    """
    Get various bits of info from the full path of a file.

    Example
    >>> get_file_info('../data/prod/train/trilobites/0263.jpg')
    File_info(base='0263.jpg', cohort='train', cname='trilobites', stem='0263', ext='.jpg')
    """
    _path, base = os.path.split(fname)    # base: the name of the file
    _path, cname = os.path.split(_path)   # cname: the class name
    _path, cohort = os.path.split(_path)  # cohort: train or val
    stem, ext = os.path.splitext(base)    # stem, ext: file stem and extension

    File_info = namedtuple('File_info', ['base', 'cohort', 'cname', 'stem', 'ext'])
    return File_info(base, cohort, cname, stem, ext)


def make_train_test(path, include=None, skip=None):
    """
    Take a POSIX path, with wildcards, and turn the image files into arrays.

    Example
    >>> path = '../data/prod/*/*/*.jpg'
    >>> X_train, X_val, y_train, y_val = make_train_test(path)
    >>> X_train.shape, y_train.shape
    ((528, 4096), (528,))
    """

    X_train, X_val, y_train, y_val = [], [], [], []

    for fname in glob.glob(path, recursive=True):
        base, cohort, cname, stem, ext = get_file_info(fname)

        if skip is None:
            skip = []

        if cname in skip:
            continue

        if (include is not None) and (cname not in include):
            continue

        im = Image.open(fname)
        img_i = np.asarray(im, dtype=np.float) / 255

        if cohort == 'train':
            X_train.append(img_i.ravel())
            y_train.append(cname)
        elif cohort == 'val':
            X_val.append(img_i.ravel())
            y_val.append(cname)

    return (np.array(X_train), np.array(X_val),
            np.array(y_train), np.array(y_val)
            )


def preprocess_images(path,
                      target,
                      size=None,
                      prop=0.25,
                      grey=True,
                      verbose=True
                      ):
    """
    Prepare the training and validation data.

    Args:
        path (str): The POSIX path, globbable.
        target (str): The name of the directory in which to put everything.
        size (tuple): Tuple of ints, the size of the output images.
        prop (float): Proportion of images to send to val (rest go to train).
        grey (bool): Whether to send to greyscale.
        verbose (bool): Whether to report out what the function did.

    Returns:
        None
    """
    trn = 0
    count = 0

    if size is None:
        size = (32, 32)

    for i, fname in enumerate(glob.glob(path)):

        # Read various bits of the path.
        path, base = os.path.split(fname)
        _, folder = os.path.split(path)
        name, ext = os.path.splitext(base)

        # Try to open the file.
        try:
            with Image.open(fname) as im:

                count += 1
                # Send 25% to val folder.
                cohort = 'train' if i % round(1/prop, 0) else 'val'

                # Form an output filename.
                outfile = os.path.join(target, cohort, folder, f'{i:04d}.jpg')

                # Resize, remove alpha, and save.
                im = im.resize(size, Image.ANTIALIAS)

                if grey:
                    im = im.convert('L')

                im.save(outfile)

                # Flip and rotate.
                if cohort == 'train':
                    trn += 1
                    outfile = os.path.join(target,
                                           cohort,
                                           folder,
                                           f'{i:04d}r.jpg')
                    im.transpose(Image.ROTATE_90).save(outfile)
                    outfile = os.path.join(target,
                                           cohort,
                                           folder,
                                           f'{i:04d}f.jpg')
                    im.transpose(Image.FLIP_LEFT_RIGHT).save(outfile)

        except OSError as e:
            print(f'{folder}/{base} rejected:', e)
            continue

    if verbose:
        print(f"Wrote {trn} files to {os.path.join(target, 'train')}")
        print(f"Wrote {count-trn} files to {os.path.join(target, 'val')}")

    return None


def plot_activation(f, domain=(-5, 5), **kwargs):
    """
    Plot an activation function and its derivative.

    kwargs are passed to the function f.
    """
    fig, ax = plt.subplots(figsize=(10, 3))

    # Plot the function.
    x = np.linspace(*domain, 100)
    y = f(x, **kwargs)
    ax.plot(x, y, color='C9', lw=3, label='f(z)')

    # Plot its derivative
    y_ = f(f(x, **kwargs), derivative=True, **kwargs)
    ax.plot(x, y_, '--', color='r', lw=3, label="f'(z)")

    mi, ma = y.min() - 0.1, y.max() + 0.1

    ax.axis('equal')
    ax.set_ylim(mi, ma)
    ax.axvline(0, color='k', lw=0.75, zorder=0)
    ax.grid(color='k', alpha=0.2)
    ax.axhline(0, c='k', alpha=0.5, lw=1.25)
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.set_frame_on(False)
    ax.tick_params(axis='both', which='both', length=0)

    plt.legend(loc='upper left')
    plt.show()

    return


def plot_confusion_matrix(y_true, y_pred, normalized=True):
    """
    The sklearn version of this function does not
    allow you to pass in a cross_validation result.
    """
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true, y_pred)
    if normalized:
        norm = (cnf_matrix.T / np.sum(cnf_matrix, axis=1)).T
    else:
        norm = cnf_matrix
    classes = np.unique(y_true)

    # Plot non-normalized confusion matrix.
    plt.figure(figsize=(6, 6))
    plt.imshow(norm, interpolation='nearest', cmap='Greens', vmin=0, vmax=1)
    plt.title("Confusion matrix")
    plt.colorbar(shrink=0.67)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Print the support numbers inside the plot.
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.ylim(5.5, -0.5)  # Bug in mpl.
    plt.tight_layout()
    plt.show()

    return


def _create_comparison_data(data=None, replace=False):
    """Make the default data used for the comparison plot.

    Args:
        data (tuple, optional): Data of the form (X, y). Defaults to None.
        replace (bool, optional): Whether to add a new dataset to the comparison plot.
                                  False will add to the default data.
                                  True will replace and use the new dataset only. Defaults to False.

    Returns:
        list of tuples: list of tuples each of the form (X, y)
    """
    if replace:
        return [data]
    X, y = make_classification(
        n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    datasets = [
        make_moons(noise=0.3, random_state=0),
        make_circles(noise=0.2, factor=0.5, random_state=1),
        linearly_separable,
    ]

    df = pd.read_csv('https://geocomp.s3.amazonaws.com/data/RPC_simple.csv')
    X = df[['Vp', 'rho']].values
    y = df.Lithology.values

    datasets += [(X, y == 'sandstone')]
    if not replace and data is not None:
        datasets += [data]

    return datasets


def make_comparison_plot(classifiers=None, data=None,
                         replace_dataset=False, replace_classifiers=False):
    """Make a plot based on https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
    This function lets you change the classifiers that are compared,
    as well as the dataset to be classified.

    The plot will be sized giving each row and column equal space, so a single
    dataset will only show a single row. The same is true for classifiers and 
    columns.

    At present, only a single dataset in addition to the four used by default
    can be passed, but any number of classifiers can be passed (in theory).

    This function should probably be split, with the plotting code in a helper
    function of its own.

    Args:
        classifiers (dict, optional): key is string to be used for title in plot
                                      value is the instantiation of a classifier.
                                      Defaults to None.
        data (tuple, optional): Needs to be of the form (X, y). Defaults to None.
        replace_dataset (bool, optional): If True, this will use the dataset supplied
                                             instead of the default.
                                          If False, this will add the dataset to the
                                             default set and plot all of them.
                                          Requires `data` to be passed.
                                          Defaults to False.
        replace_classifiers (bool, optional): If True, this will use the classifiers
                                                passed instead of the default selection.
                                              If False, this will add the classifiers
                                                passed to the default selection.
                                              Requires `classifiers` to be passed.
                                              Defaults to False.
    Returns:
        None, but a plot is made as a side-effect.

    Examples:
        Default plot, no changes.
        >>> make_comparison_plot()

        Use only new classifiers, default datasets.
        >>> classifiers = {
        ... 'Extra Trees'              : ExtraTreesClassifier(),
        ... 'Decision Tree\n(depth=3)' : DecisionTreeClassifier(max_depth=3),
        ... 'Decision Tree\n(depth=10)': DecisionTreeClassifier(max_depth=10),
        ... }
        >>> make_comparison_plot(classifiers=classifiers, replace_classifiers=True)

        Use only new data, default classifiers.
        >>> df = pd.read_csv('https://geocomp.s3.amazonaws.com/data/RPC_simple.csv')
        >>> X = df[['Vp', 'rho']].values
        >>> y = df.Lithology.values
        >>> data = (X, y == 'sandstone') # we need to add a tuple of (X, y)
        >>> make_comparison_plot(data=data, replace_dataset=True)
    """
    if classifiers:
        classifiers_new = classifiers.copy()
        # We need a copy because otherwise we are changing the original dict.
    else:
        classifiers_new = classifiers
    
    # We will use this if we do not already have any classifiers.
    classifiers_default = {
        "Nearest Neighbors\n(k=1)": KNeighborsClassifier(1),
        "Nearest Neighbors\n(k=9)": KNeighborsClassifier(9),
        "Linear SVM": SVC(kernel='linear'),
        "RBF SVM\n(C=1)": SVC(gamma=2, C=1),
        "RBF SVM\n(C=10)": SVC(gamma=2, C=10),
        "Gaussian Process": GaussianProcessClassifier(1.0 * RBF(1.0)),
        "Decision Tree\n(depth=2)": DecisionTreeClassifier(max_depth=2),
        "Decision Tree\n(depth=15)": DecisionTreeClassifier(max_depth=15),
        "Random Forest\n(depth=2)": RandomForestClassifier(max_depth=2),
        "Neural Network": MLPClassifier(alpha=1, max_iter=1000),
    }

    if classifiers_new is not None and replace_classifiers is True:
        # We have classifiers and want to replace the defaults.
        # We do not actually need this case, but explicit is good.
        pass
    elif classifiers_new is not None and replace_classifiers is False:
        # We have classifiers and want to add to the defaults.
        classifiers_new.update(classifiers_default)
    else:
        # We have not been given classifiers, so use the defaults.
        classifiers_new = classifiers_default

    h = 0.02  # step size in the mesh

    datasets = _create_comparison_data(data, replace=replace_dataset)

    height, width = 2*len(datasets), 2*len(classifiers_new) + 2

    figure = plt.figure(figsize=(width, height))
    i = 1
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=42
        )

        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # just plot the dataset first
        colors = ['#7c91b9', '#e5e8eb', '#74cada']
        cm = LinearSegmentedColormap.from_list("mycmap", colors)
        cm_bright = ListedColormap(["#7c91b9", "#74cada"])
        ax = plt.subplot(len(datasets), len(classifiers_new) + 1, i)
        edgecolor1 = '0.4'
        edgecolor2 = '0.2'
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors=edgecolor1)
        # Plot the testing points
        ax.scatter(
            X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors=edgecolor2
        )
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in classifiers_new.items():
            ax = plt.subplot(len(datasets), len(classifiers_new) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=0.8)

            # Plot the training points
            ax.scatter(
                X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors=edgecolor1
            )
            # Plot the testing points
            ax.scatter(
                X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                edgecolors=edgecolor2, alpha=0.6,
            )

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)
            ax.text(
                xx.max() - 0.3,
                yy.min() + 0.3,
                ("%.2f" % score).lstrip("0"),
                size=15,
                horizontalalignment="right",
            )
            i += 1

    plt.tight_layout()
    plt.show()
    return 


def plot_ribbons(*arrs, s=None, legend=None, cmap=None, classes=None, titles=None, ticks=None):
    """
    Plot one or more 1D arrays, e.g. of class labels, as ribbons.
    The classes must be integers, eg transformed using LabelEncoder.

    Args:
        arrs: One or more 1D arrays of class labels.
        s (tuple of ints): Optional slice tuple, e.g. `(None, 100)` to plot
            only the first 100 samples.
        legend (dict): A dictionary mapping class labels (natural names as
            strings) to colours (eg hex strings).
        cmap (str): A matplotlib colormap name.
        classes (list): A list of class labels (natural names as strings).
            Don't pass this if you pass a legend.
        titles (list): A list of titles for the plots. By default, 2 arrs will
            be labelled 'Actual' and 'Predicted'.
        ticks (tuple): (minor, major) tick intervals on the y-axis.

    Returns:
        None.
    """
    if s is not None:
        s = slice(*s)
        
    if legend and (classes is not None):
        raise ValueError('Class names will be drawn from the legend; do not pass both. Set classes=None.')
        
    if legend is not None:
        class_enc = np.arange(len(legend))
    else:
        class_enc = np.unique(np.hstack([arr[s] for arr in arrs]))

    mi, ma = class_enc[[0, -1]]
    rng = ma - mi + 1

    if legend is not None:
        cmap = ListedColormap(legend.values(), 'indexed')
        classes = list(legend.keys())
    else:
        if isinstance(cmap, str):
            cmap = get_cmap(cmap)
        else:
            cmap = cmap or plt.cm.viridis
        colours = [cmap(i) for i in np.linspace(0, 1, rng)]
        cmap = ListedColormap(colours, 'indexed')
        
        classes = classes or class_enc
        
    if len(arrs) == 1:
        titles = titles or ['Labels']
    elif len(arrs) == 2:
        titles = titles or ['Actual', 'Predicted']
    else:
        titles = titles or ['Actual'] + [f'Pred {n}' for n in range(len(arrs)-1)]
        
    fig, axs = plt.subplots(ncols=len(arrs), figsize=(2*len(arrs)+1, 12), sharey=True)

    for ax, name, arr in zip(axs, titles, arrs):
        im = ax.imshow(arr[s].reshape(-1, 1), aspect='auto', cmap=cmap, vmin=mi-0.5, vmax=ma+0.5, interpolation='none')
        ax.set_title(name)
        ax.xaxis.set_visible(False)
        
        if ticks is not None:
            minor, major = ticks
            if ax.get_subplotspec().is_first_col():
                loc = plticker.MultipleLocator(base=minor)
                ax.yaxis.set_minor_locator(loc)
                loc = plticker.MultipleLocator(base=major)
                ax.yaxis.set_major_locator(loc)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.5])
    cbar = fig.colorbar(im, ticks=class_enc, cax=cbar_ax)
    cbar.ax.set_yticklabels(classes)
    
    return None


def logistic_plots(model, X_val, y_val, y_test, cutoff):
    """
    Docstring
    """
    prob_shale = model.predict_proba(X_val)[:,1]
    predicted = [1 if i > cutoff else 0 for i in prob_shale]  #Setting a cutoff

    data = [[X_val, y_val=='shale', y_val=='shale'],
            [X_val, y_test, y_val=='shale'],
            [X_val, y_test, model.predict_proba(X_val)[:,1]], 
            [X_val, y_test, predicted]]

    mi, ma = np.amin(X_val) * 0.9, np.amax(X_val) * 1.1
    ymi, yma = -0.05, 1.05 
    titles = ['Data', 'Logistic Regression', 'Probabilty of shale', 'Classification']
    cmap = plt.get_cmap('viridis')
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    cax = fig.add_axes([0.665, 0.55, 0.005, 0.3])
    for d, ax, t in zip(data, axs, titles):
        im = ax.scatter(d[0], d[1], c=d[2], ec='grey', s=50, alpha=0.5)
        left, bottom, width, height = (mi, cutoff, ma-mi, yma - cutoff )
        rect_top = plt.Rectangle((left, bottom), width, height, alpha=0.1, facecolor=cmap(1.0))
        rect_bot = plt.Rectangle((left, ymi), width, cutoff-ymi, alpha=0.1, facecolor=cmap(0.0))
        ax.add_patch(rect_top)
        ax.add_patch(rect_bot)
        ax.set_xlim(mi, ma)
        ax.set_ylim(ymi, yma)
        ax.set_title(t)
        ax.text(np.amin(X_val), cutoff, s=f'cutoff: {cutoff}', va='bottom', ha='left')
        ax.set_xlabel('Vp [m/s]')
        ax.axhline(cutoff, c='grey')
        
    fig.colorbar(im, cax=cax, orientation='vertical')
    return


def lithology_tree(clf, features, figsize=(15, 8)):
    """"Plots a decision tree using a lithologic-friendly set of colors
    defined by the colordict dictionary"""
    
    colordict = {'dolomite': 'blueviolet', 
                  'limestone': 'cornflowerblue',
                  'sandstone': 'goldenrod',
                  'shale': 'darkseagreen'}

    fig, ax = plt.subplots(figsize=figsize)
    artists = tree.plot_tree(clf, feature_names=features, class_names=clf.classes_,
                             filled=True, rounded=True, ax=ax)
    
    colors = list(colordict.values())

    for artist, impurity, value in zip(artists, clf.tree_.impurity, clf.tree_.value):
        # let the max value decide the color; whiten the color depending on impurity (gini)
        
        r, g, b = to_rgb(colors[np.argmax(value)])
        f = impurity  # for N colors: f = impurity * N/(N-1) if N>1 else 0
        facecolor = (f + (1-f)*r, f + (1-f)*g, f + (1-f)*b)
        artist.get_bbox_patch().set_facecolor(facecolor)
        artist.get_bbox_patch().set_edgecolor('black')

    return


def show_decision_regions(clf,
                          X_train, y_train, X_val, y_val,
                          palette=None, hue_order=None,
                          plot_train=False, ax=None,
                          scaler=None,
                          ):
    """
    Plot decision boundaries in a multi-class problem with two features.
    
    Args:
        clf: The classifier.
        X_train (array): The training input.
        y_train (array): The training labels.
        X_val (array): The validation data.
        y_val (array): The validation labels.
        palette: The Seaborn palette.
        hue_order: The hue order for Seaborn.
        plot_train (bool): Whether to plot the training data.
        ax (Axes): An Axes object. If None, one will be created.
        scaler (sklearn scaling transformer): The scaler; must have a 

    Returns:
        Axes.
    """
    lithologies = {'sandstone': 1, 'shale': 2, 'limestone': 3, 'dolomite': 4}
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,10))

    if scaler is not None:
        X_train = scaler.inverse_transform(X_train)
        X_val = scaler.inverse_transform(X_val)

    if plot_train:
        _ = sns.scatterplot(x=X_train[:,0], y=X_train[:,1], hue=y_train, s=200, ec='none', alpha=0.25, 
                            palette=palette, hue_order=hue_order, 
                            ax=ax, legend=False)

    ax = validation_scatter(X_val, y_val, y_pred, palette, hue_order, ax=ax)

    # Plot the decision boundary.
    h = .02  # step size in the mesh
    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.asarray([lithologies[d] for d in Z])

    title = f"{str(clf).split('(')[0]}   Accuracy: {accuracy_score(y_val, y_pred):.3f}"
    ax.set_title(title, fontsize=14)
    
    # Put the result into a color plot.
    Z = Z.reshape(xx.shape)
    im = ax.pcolormesh(xx, yy, Z, alpha=0.2, zorder=1, shading='auto',
                       cmap=ListedColormap(colors=['goldenrod', 'darkseagreen', 'cornflowerblue', 'blueviolet']))

    return ax

def validation_scatter(X_val, y_val, y_pred, palette=None, hue_order=None, ax=None, scaler=None):
    """
    Plot validation points X_val, y_val in comparison to y_pred.
    Validation points are the large dots. Predictions are the small dots.
    Works only with the Classification_algorithms notebook.

    Args:
        X_val (array): The validation data.
        y_val (array): The validation labels.
        y_pred (array): The predicted labels.
        palette: The Seaborn palette.
        hue_order: The hue order for Seaborn.
        ax (Axes): An Axes object. If None, one will be created.

    Returns:
        Axes.

    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))

    if scaler is not None:
        X_val = scaler.inverse_transform(X_val)

    # Plot the validation data
    scatter = sns.scatterplot(x=X_val[:,0], y=X_val[:,1], hue=y_val, s=200, ec='k', alpha=0.5, 
                              palette=palette, hue_order=hue_order, ax=ax)

    # Plot the predicted classes
    _ = sns.scatterplot(x=X_val[:,0], y=X_val[:,1], hue=y_pred, s=50, ec='k', alpha=0.75, 
                        palette=palette, hue_order=hue_order, 
                        ax=ax, legend=False)
    return ax


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
