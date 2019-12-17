from functools import partial
from skimage import exposure
import numpy as np
from . import msrcr


def contrast_stretching(img, **junk):
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))


def hist_eq(img, focus_region, **junk):
    return exposure.equalize_hist(img, mask=focus_region)


def adaptive_hist_eq(img, **junk):
    return exposure.equalize_adapthist(img, clip_limit=0.03)


def msrcr_retinex(img, focus_region, **junk):
    img = np.ma.masked_array(img*255, ~focus_region)
    im_out = np.array(msrcr.MSRCR(img, 60, 3))/255
    return im_out
