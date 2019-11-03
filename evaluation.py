"""
Compare various methods
"""
import cv2
import numpy as np
from skimage import color
import matplotlib.pyplot as plt #importing matplotlib
from scipy import stats
from skimage import exposure
import skimage
import pandas as pd

import dehaze
import metric
from idrid import IDRiD
import util


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
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def test_compare_hist_methods_mixture_of_gaussians():
    x = np.random.randn(1000)
    pd.DataFrame({'method %s' % method: [metric.compare_hist(
        x/2+(np.random.randn(1000)*1+i)/2, np.random.randn(1000)*1 + i, method)
        for i in range(20)] for method in [0, 1]}
    ).plot()


if __name__ == "__main__":

    dset = IDRiD('./data/IDRiD_segmentation')

    img, labels = dset['IDRiD_03']
    he = labels['HE']
    ma = labels['MA']
    ex = labels['EX']
    se = labels['SE']
    od = labels['OD']

    # set background pure black.
    bg = util.get_background(img)
    img[bg] = 0

    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    img_cs = exposure.rescale_intensity(img, in_range=(p2, p98))
    # Equalization
    img_eq = exposure.equalize_hist(img)
    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)

    # plot the result of the methods
    f, axs = plt.subplots(2, 2)
    for img, ax in zip([img, img_cs, img_eq, img_adapteq], axs.ravel()):
        ax.imshow(img)

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
    plt.show()
