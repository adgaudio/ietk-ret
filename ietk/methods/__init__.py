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



import numpy as np
import scipy.ndimage as ndi
from cv2.ximgproc import guidedFilter


def solvet(I, A, use_gf=True, fsize=(5,5)):
    z = 1-ndi.minimum_filter((I/A).min(-1), fsize)
    if use_gf:
        z = gf(I, z)
    rv = z.reshape(*I.shape[:2], 1)
    return rv


def gf(guide, src, r=100, eps=1e-8):
    return guidedFilter(guide.astype('float32'), src.astype('float32'), r, eps).astype('float64')
def solveJ(I, A, t):
    epsilon = max(np.min(t)/2, 1e-8)
    return (I-A)/np.maximum(t, epsilon) + A
def solvet(I, A, use_gf=True, fsize=(5,5)):
    z = 1-ndi.minimum_filter((I/A).min(-1), fsize)
    if use_gf:
        z = gf(I, z)
    rv = z.reshape(*I.shape[:2], 1)
    return rv
def ta(img):
    return solvet(1-img, 1)
def td(img):
    return 1-solvet(1-img, 1)
def tb(img):
    I = img.copy()
    I[:,:,2] = 1
    return solvet(I, 1)
def tc(img):
    I = img.copy()
    I[:,:,2] = 1
    return 1-solvet(I, 1)

all_methods = {
    'identity': lambda img, focus_region, **kwargs: img,
    'A': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, ta(img)), ~focus_region),
    'A2': lambda img, focus_region, **kwargs: solveJ(img, 0, ta(img)),
    'B': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, tb(img)), ~focus_region),
    'C': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, tc(img)), ~focus_region),
    'C2': lambda img, focus_region, **kwargs: sharpen_img.sharpen(sharpen_img.sharpen(solveJ(img, 0, tc(img)), ~focus_region), ~focus_region),
    'C3': lambda img, focus_region, **kwargs: sharpen_img.sharpen(sharpen_img.sharpen(sharpen_img.sharpen(solveJ(img, 0, tc(img)), ~focus_region), ~focus_region), ~focus_region),
    'D': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, td(img)), ~focus_region),
    'W': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 1, ta(img)), ~focus_region),
    'X': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 1, tb(img)), ~focus_region),
    'Y': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 1, tc(img)), ~focus_region),
    'Z': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 1, td(img)), ~focus_region),
    r'A+X': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, ta(img))/2 + solveJ(img, 1, tb(img))/2, ~focus_region),
    r'C+X': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, tc(img))/2 + solveJ(img, 1, tb(img))/2, ~focus_region),
    r'A+C': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, ta(img))/2 + solveJ(img, 0, tc(img))/2, ~focus_region),
    r'A+Z': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, ta(img))/2 + solveJ(img, 0, td(img))/2, ~focus_region),
    r'A+C+X': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, ta(img))/3 + solveJ(img, 0, tc(img))/3 + solveJ(img, 1, tb(img))/3, ~focus_region),
    r'A+C+X+Z': lambda img, focus_region, **kwargs: sharpen_img.sharpen(solveJ(img, 0, ta(img))/4 + solveJ(img, 0, tc(img))/4 + solveJ(img, 1, tb(img))/4 + solveJ(img, 1, td(img))/4, ~focus_region),
}
def A(img):
    return solveJ(img, 0, ta(img))
def B(img):
    return solveJ(img, 0, tb(img))
def C(img):
    return solveJ(img, 0, tc(img))
def X(img):
    return solveJ(img, 1, tb(img))

_m = all_methods
all_methods.update({
    'A+B': lambda img, focus_region: sharpen_img.sharpen(A(img)/2 + B(img)/2, ~focus_region),
    'sA+sB': lambda img, focus_region: _m['A'](img=img,focus_region=focus_region)/2 + _m['B'](img=img,focus_region=focus_region)/2,
    #  'B+C': lambda *a, **k: _m['B'](*a, **k)/2 + _m['C'](*a, **k)/2,
    #  'B+X': lambda *a, **k: _m['B'](*a, **k)/2 + _m['X'](*a, **k)/2,
    #  'A+B+C': lambda *a, **k: _m['A'](*a, **k)/3 + _m['B'](*a, **k)/3 + _m['C'](*a, **k)/3,
    #  'B+C+X': lambda *a, **k: _m['B'](*a, **k)/3 + _m['C'](*a, **k)/3 + _m['X'](*a, **k)/3,
    #  'A+B+C+X': lambda *a, **k: _m['A'](*a, **k)/2 + _m['B'](*a, **k)/2 + _m['C'](*a, **k)/2 + _m['X'](*a, **k)/2,
})


#  all_methods = {
    #  # method_name: func
    #  'Unmodified Image': identity,
    #  'Dehazed (DCP)': dehaze_dcp,
    #  'Illuminated (DCP)': illuminate_dcp,
    #  'Illuminated-Dehazed (DCP)': illuminate_dehaze_dcp,
    #  'Sharpen, t=0.15': sharpen,
    #  'Illuminate Sharpen': illuminate_sharpen,
    #  'MSRCR (Retinex)': msrcr_retinex,
    #  #  'Bayes Sharpen, t>=0.15': bayes_sharpen,
    #  'Contrast Stretching': contrast_stretching,
    #  'Histogram Eq.': hist_eq,
    #  'Adaptive Histogram Eq.': adaptive_hist_eq,
#  }
