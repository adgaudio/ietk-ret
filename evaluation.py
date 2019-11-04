"""
Compare various methods
"""
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import skimage

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
    f, axs = plt.subplots(2, 4)
    for ax in axs.ravel():
        ax.axis('off')
    for (method_name, img), ax in zip(imgs_denoised.items(), axs.ravel()):
        ax.imshow(img)
        ax.set_title(method_name, {'fontsize': 'small'})
    return f


def _evaluation_args(imgs_denoised, labels, evaluation_func):
    for method_name, modified_img in imgs_denoised.items():
        for lesion_type, mask in labels.items():
            for stat_name, func in evaluation_func.items():
                yield (method_name, modified_img, lesion_type, mask, stat_name,
                       func)


def _evaluate(method_name, modified_img, lesion_type, mask, stat_name, func):
    return {
        'Method': method_name,
        'Evaluation Method': stat_name,
        'Statistic': func(modified_img, mask),
        'Lesion': lesion_type}


def empirical_evaluation(
        imgs_denoised, labels, evaluation_func,
        plot=True, num_procs=None):
    """Given a set of images, evaluate how well each one separates
    the healthy from diseased pixels for a set of lesion types.

    Input:
        imgs_denoised - dict[str, array] of form {method_name: output_img}
            Images output from the set of methods we wish to evaluate
        labels - dict[str,bool_array] of form {lesion_name: label_mask}
            Boolean masks containing ground truth labels (ie lesion annotation)
        evaluation_func - (func|dict[str,func])
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
    args = _evaluation_args(imgs_denoised, labels, evaluation_func)
    if num_procs == 1:
        data = [_evaluate(*x) for x in args]
    else:
        pool = mp.Pool(num_procs)
        data = pool.starmap(_evaluate, args)
        del pool
    # format into nice result and plot
    df = pd.DataFrame(data)
    if plot:
        fig = sns.catplot(
            x='Lesion', y='Statistic', hue='Method', col='Evaluation Method',
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
    bg = util.get_background(img)
    img[bg] = 0

    # set location to save data
    os.makedirs(results_dir, exist_ok=True)

    # get the denoised images for all competing methods
    pool = mp.Pool(num_procs)
    rv = [(name, pool.apply_async(func, [img]))
          for name, func in competing_methods.all_methods.items()]
    imgs_denoised = {name: res.get() for name, res in rv}
    del pool, rv

    # generate plots
    if qualitative:
        fig = plot_qualitative(imgs_denoised)
        fig.savefig(os.path.join(results_dir, 'qualitative_%s.png' % img_id))
    if empirical:
        df, fig2 = empirical_evaluation(
            imgs_denoised, labels, metric.eval_methods.copy(),
            num_procs=num_procs)
        fig2.savefig(os.path.join(results_dir, 'empirical_%s.png' % img_id))
        df.to_csv(
            os.path.join(results_dir, 'stats_%s.csv' % img_id),
            index=False)
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
