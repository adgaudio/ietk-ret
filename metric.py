import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


def ks_test_max_per_channel(img, mask, focus_region):
    """Compute a 2-sample Kolmogorov-Smirnov statistic on each channel of image
    returning the max value across channels.  Ignore the background regions

    img - array of shape (x, y, z) with z being the channels.
    mask - bool array separating healthy from diseased pixels
    focus_region - bool array containing the area to not ignore.

    Return float in [0, 1]
        0 if the healthy and diseased pixels have the same distribution,
        1 if they come from different distribution.

    Input:
        img - array of shape (h,w,ch)
            Input image, which may have a variable number of channels
        mask - boolean array of shape (h, w)
            Ground truth labels identifying healthy vs diseased pixels.
    """
    bg = ~focus_region
    assert mask.dtype == 'bool'
    assert mask.shape == img.shape[:2]
    maxstat = 0
    for ch in range(img.shape[2]):
        # get diseased (a) and healthy (b) pixels
        a = img[:, :, ch][mask & (~bg[:, :, ch])]
        b = img[:, :, ch][(~mask) & (~bg[:, :, ch])]
        # compute the stat
        statistic = stats.ks_2samp(a.ravel(), b.ravel())[0]
        maxstat = max(statistic, maxstat)
    return maxstat


eval_methods = {
    'KS Test, max of the channels': ks_test_max_per_channel  # test if healthy separable from diseased.
    # need test if lesions (MA,HE,etc) have the same intensities across images.  how about:  var(sum_imgs(p(rgb|lesion=MA)))  where p(rgb|lesion) is a 256*3 vector for each img.
}

def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

# Functions to comare histograms
def compare_hist(x1,x2,num):
    if(num==0):
        # Kolmogorov–Smirnov test
        D,p = stats.ks_2samp(x1.ravel(), x2.ravel())
    elif(num==1):
        # Basically, it calculates the overlap between the two histograms and
        # then normalizes it by second histogram (you can use first).
        # You can use this function to calculate the similarity between the histograms.
        hist_1, _ = np.histogram(x1, bins=100, range=[0, 256])
        hist_2, _ = np.histogram(x2, bins=100, range=[0, 256])
        D = return_intersection(hist_1,hist_2)

    return D

# Give any imoriginal used after removing noise,
# imbinary images are labelled images from director "dir2" defined later
# Will return 2 arrays, i.e an array of pixel values which are good part of eye
# And onther array of pixels which represents bad part of eye
def get_good_bad_pixels(imoriginal,imbinary):
    #  imbinary is True if diseased, False if healthy.


    bad_pixel = imoriginal[imbinary != 0]
    good_pixel = imoriginal[imbinary == 0]
    #  good_pixel = good_pixel[good_pixel > 15]

    return bad_pixel, good_pixel


def test_compare_hist_methods_mixture_of_gaussians():
    x = np.random.randn(1000)
    pd.DataFrame({'method %s' % method: [metric.compare_hist(
        x/2+(np.random.randn(1000)*1+i)/2, np.random.randn(1000)*1 + i, method)
        for i in range(20)] for method in [0, 1]}
    ).plot()


if __name__ == "__main__":
    dir = '/home/ankit/Desktop/MLSP/project/indian-diabetic-retinopathy-dataset/indian-diabetic-retinopathy-dataset/A. Segmentation/1. Original Images/a. Training Set/'
    dir2 = '/home/ankit/Desktop/MLSP/project/indian-diabetic-retinopathy-dataset/indian-diabetic-retinopathy-dataset/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/2. Haemorrhages/'

    file = 'IDRiD_02'
    imoriginal = cv2.imread(dir+file+'.jpg')
    imbinary = cv2.imread(dir2+file+'_HE.tif')[:, :, 2]

    for ch in [0, 1, 2]:
        bad_pixel, good_pixel = get_good_bad_pixels(imoriginal[:, :, ch],imbinary)

        x = plt.hist(bad_pixel,bins=256)
        plt.xlim(0, 80) ; plt.title('bad pixel, channel: %s' % ch)
        plt.show()
        x = plt.hist(good_pixel,bins=256)
        plt.xlim(0, 80) ; plt.title('good pixel, channel: %s' % ch)
        plt.show()

        # Function to compare bad pixels with good pixels using metic defined
        metric = compare_hist(bad_pixel, good_pixel, 1)

        print(metric,"ch ",ch)
