#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import random
import itertools
from collections import namedtuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.stats import mode


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


if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
