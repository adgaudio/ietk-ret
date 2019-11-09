"""
Compare various methods
"""
from collections import namedtuple
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import skimage
from functools import partial

import metric
from idrid import IDRiD
import util
import competing_methods


def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.
    """
    image = skimage.img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = skimage.exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def plot_qualitative(imgs_denoised):
    f, axs = plt.subplots(2, 4, figsize=(12, 8))
    for ax in axs.ravel():
        ax.axis('off')
    for (method_name, img), ax in zip(imgs_denoised.items(), axs.ravel()):
        ax.imshow(img)
        ax.set_title(method_name)
    f.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=.01, hspace=-.4)
    return f


_Zargs = namedtuple('_Zargs', [
    'get_img_and_labels', 'get_focus_region', 'method_name', 'func',
    'label_name', 'stat_name', 'eval_func'])

def _evaluation_args(get_img_and_labels, get_focus_region,
                     label_names, methods, evaluation_funcs):
    for method_name, func in methods.items():
        for label_name in label_names:
            for stat_name, eval_func in evaluation_funcs.items():
                yield _Zargs(
                    get_img_and_labels, get_focus_region, method_name,
                    func, label_name, stat_name, eval_func)


def _evaluate(Z):
    modified_img, label_mask, focus_region = _evaluate_func(Z)
    return {
        'Method': Z.method_name,
        'Evaluation Method': Z.stat_name,
        'Statistic': Z.eval_func(
            modified_img, label_mask, focus_region),
        'Class': Z.label_name}


def _evaluate_func(Z, _ensure_labels=True):
    img, labels = Z.get_img_and_labels()
    try:
        label_mask = labels[Z.label_name]
    except:
        if _ensure_labels:
            return  # the label is not available for this image
    focus_region = Z.get_focus_region(img)

    # center image
    img = util.zero_mean(img, focus_region)

    modified_img = Z.func(img=img, focus_region=focus_region, **Z._asdict())
    return modified_img, label_mask, focus_region


def empirical_evaluation(
        get_img_and_labels, get_focus_region, label_names, methods,
        evaluation_funcs, plot=True, num_procs=None):
    """For a given image,
    evaluate how well each method separates the foreground from background
    pixels of each given "label" mask, using one or more evaluation methods.

    Input:
        get_img_and_labels - func
            A serializable function with no inputs that should return a tuple
            containing an image and a dict of label masks.  Signature:
                f() -> (img, {'label1': binary_foreground_mask})
            The function must support `pickle.dumps(get_img)` if num_procs!=1
        get_focus_region - func
            A serializable function that should receive as input an image
            and return the focus region.
            The function must support `pickle.dumps` if num_procs!=1
        label_names - list[str]
            List the labels to evaluate.
            Assume are created by `get_img_and_labels`.  Assume that all labels
            might not be available for all imgs.
        methods - dict[str,func] of form {method_name: f}
            Each function is expected to return an image and has signature:
                f(img, *, focus_region, label_name) -> newimg
                The focus_region is a binary mask which specifies which pixels
                in the input img should use (1) or ignore (0).
        evaluation_funcs - dict[str,func]
            Func expects as input two arrays and outputs a similarity score
        plot - bool
            If False, don't plot.
            Otherwise, just let
        num_procs - (int)
            Number of multiprocessing workers to use to parallelize task.
    Output: tuple
        pandas.DataFrame in long format - the score for each denoised image
            for each label for each evaluation func
        seaborn.FacetGrid - the plot that was generated or None
    """
    # Compute the statistics on the given data.
    args = _evaluation_args( get_img_and_labels, get_focus_region, label_names, methods, evaluation_funcs)
    if num_procs == 1:
        data = [_evaluate(x) for x in args]
    else:
        with mp.Pool(num_procs) as pool:
            data = pool.map(_evaluate, args)
    # format into nice result and plot
    df = pd.DataFrame(x for x in data if x is not None)
    if plot:
        fig = sns.catplot(
            x='Class', y='Statistic', hue='Method', col='Evaluation Method',
            data=df, kind='bar')
    else:
        fig = None
    return df, fig


def evaluate_idrid_img(
        img_id, qualitative=True, empirical=True,
        results_dir='./data/evaluation_idrid', num_procs=None):
    dset = IDRiD('./data/IDRiD_segmentation')
    img, labels = dset.load_img(img_id)

    # set background pure black.
    #  bg = util.get_background(img)
    #  img[bg] = 0

    # set location to save data
    os.makedirs(results_dir, exist_ok=True)
    get_img_and_labels = partial(dset.load_img, img_id)
    get_focus_region = util.get_foreground
    label_names = ['MA', 'HE', 'EX', 'SE', 'OD']

    if empirical:
        df, fig2 = empirical_evaluation(
            get_img_and_labels, get_focus_region, label_names,
            competing_methods.all_methods.copy(), metric.eval_methods.copy(),
            num_procs=num_procs)
        fig2.savefig(os.path.join(results_dir, 'empirical_%s.png' % img_id))
        df.to_csv(
            os.path.join(results_dir, 'stats_%s.csv' % img_id),
            index=False)

    if qualitative:
        args = _evaluation_args(
            get_img_and_labels, get_focus_region, label_names,
            competing_methods.all_methods.copy(), {'ignore': 'ignore'})
        if num_procs == 1:
            imgs_denoised = {  # method_name: img
                tup[2]: _evaluate_func(tup, _ensure_labels=False)[0]
                for tup in args}
        else:
            with mp.Pool(num_procs) as pool:
                rv = [(tup[2], pool.apply_async(
                    _evaluate_func, [tup], dict(_ensure_labels=False)))
                    for tup in args]
                imgs_denoised = {name: res.get()[0] for name, res in rv}
        fig = plot_qualitative(imgs_denoised)
        fig.savefig(os.path.join(results_dir, 'qualitative_%s.png' % img_id))
    return locals()


    #  for ch in [0, 1, 2]:
        #  bad_pixel, good_pixel = get_good_bad_pixels(imoriginal[:, :, ch],imbinary)

        #  x = plt.hist(bad_pixel,bins=256)
        #  plt.xlim(0, 80) ; plt.title('bad pixel, channel: %s' % ch)
        #  plt.show()
        #  x = plt.hist(good_pixel,bins=256)
        #  plt.xlim(0, 80) ; plt.title('good pixel, channel: %s' % ch)
        #  plt.show()

        #  # Function to compare bad pixels with good pixels using metic defined
        #  metric = compare_hist(bad_pixel, good_pixel, 1)

        #  print(metric,"ch ",ch)


if __name__ == "__main__":
    # run this from shell.  -j 2 requires 20gb of ram.  -j 1 requires 10gb.
    # seq -f '%02g' 1 54 | parallel -j 2 python evaluation.py 'IDRiD_{}'

    # temp hack command-line parameters
    import sys
    try:
        IMG_ID = sys.argv[1]
    except:
        IMG_ID = 'IDRiD_03'
    try:
        NUM_PROCS = int(sys.argv[2])
    except:
        NUM_PROCS = None

    evaluate_idrid_img(IMG_ID, num_procs=NUM_PROCS)
    #plt.show()

    # debugging ... here's an image ready to go
    #  dset = IDRiD('./data/IDRiD_segmentation')
    #  img, labels = dset['IDRiD_03']
    #  #  he = labels['HE']
    #  #  ma = labels['MA']
    #  #  ex = labels['EX']
    #  #  se = labels['SE']
    #  #  od = labels['OD']
    #  # set background pure black.
    #  bg = util.get_background(img)
    #  img[bg] = 0
