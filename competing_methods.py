from functools import partial
from skimage import exposure
import dehaze
import numpy as np
from sharpen_img import sharpen


def contrast_stretching(img):
    # Contrast stretching
    p2, p98 = np.percentile(img, (2, 98))
    return exposure.rescale_intensity(img, in_range=(p2, p98))


hist_eq = exposure.equalize_hist


adaptive_hist_eq = partial(exposure.equalize_adapthist, clip_limit=0.03)


def dehaze_dcp(img):
    return dehaze.dehaze(img)['radiance']


def illuminate_dcp(img):
    # Illumination via Inverted Dehazing
    return dehaze.illumination_correction(img)['radiance']


def illuminate_dehaze_dcp(img):
    # Illumination via Inverted Dehazing followed by dehazing
    return dehaze.illuminate_dehaze(img)[1]['radiance']


def identity(img):
    return img


all_methods = {
    # method_name: func
    'Unmodified Image': identity,
    'Dehazed (DCP)': dehaze_dcp,
    'Illuminated (DCP)': illuminate_dcp,
    'Illuminated-Dehazed (DCP)': illuminate_dehaze_dcp,
    'Sharpen, t=0.1': sharpen,
    'Contrast Stretching': contrast_stretching,
    'Histogram Eq.': hist_eq,
    'Adaptive Histogram Eq.': adaptive_hist_eq,
}
