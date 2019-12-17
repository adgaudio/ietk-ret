from .competing_methods import *
from . import dehaze
from . import sharpen_img
#  from . import bayes_prior

def dehaze_dcp(img, **junk):
    return dehaze.dehaze(img)['radiance']


def illuminate_dcp(img, **junk):
    # Illumination via Inverted Dehazing
    return dehaze.illumination_correction(img)['radiance']


def illuminate_dehaze_dcp(img, **junk):
    # Illumination via Inverted Dehazing followed by dehazing
    return dehaze.illuminate_dehaze(img)[1]['radiance']


def identity(img, **junk):
    return img


def sharpen(img, focus_region, **junk):
    return sharpen_img.sharpen(img, ~focus_region[:, :, 0], t=0.15)


def illuminate_sharpen(img, **kws):
    img = illuminate_dcp(img, **kws)
    return sharpen(img, **kws)


#  def bayes_sharpen(img, focus_region, label_name, **junk):
#      return bayes_prior.bayes_sharpen(
#          img, label_name, focus_region=focus_region[:, :, 0])


all_methods = {
    # method_name: func
    'Unmodified Image': identity,
    'Dehazed (DCP)': dehaze_dcp,
    'Illuminated (DCP)': illuminate_dcp,
    'Illuminated-Dehazed (DCP)': illuminate_dehaze_dcp,
    'Sharpen, t=0.15': sharpen,
    'Illuminate Sharpen': illuminate_sharpen,
    'MSRCR (Retinex)': msrcr_retinex,
    #  'Bayes Sharpen, t>=0.15': bayes_sharpen,
    'Contrast Stretching': contrast_stretching,
    'Histogram Eq.': hist_eq,
    'Adaptive Histogram Eq.': adaptive_hist_eq,
}
