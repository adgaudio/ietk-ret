import numpy as np
import scipy.ndimage as ndi


def get_background(img):
    """
    Given a retinal fundus image with a black background (assuming the
    black may not be pure black)

    Return binary mask image representing the black background.
    For each pixel in mask, True if is background, False otherwise.

    Method: computing binary closing channel-wise over img, restore
    padding lost from closing, then considering background only if all 3
    channels assumed pixel was background.
    """
    # this probably works just as well.
    #  return ~ndi.binary_opening(ndi.minimum_filter(
    #      img.min(-1), 6) > 0, np.ones((5,5)))

    img = img/img.max()
    background = (img < 20/255)
    background = ndi.morphology.binary_closing(
        background, np.ones((5, 5, 1)))
    background |= np.pad(np.zeros(
        (background.shape[0]-6, background.shape[1]-6, 3), dtype='bool'),
        [(3, 3), (3, 3), (0, 0)], 'constant', constant_values=1)
    return np.dstack([(background.sum(2) == 3)] * 3)


def get_foreground(img):
    return ~get_background(img)


def zero_mean(img, fg):
    z = img[fg]
    return (img - z.mean()) + 0.5


def norm01(img, background=None):
    """normalize in [0,1] using global min and max.
    If background mask given, exclude it from normalization."""
    if background is not None:
        tmp = img[~background]
        min_, max_ = tmp.min(), tmp.max()
    else:
        min_, max_ = img.min(), img.max()
    rv = (img - min_) / (max_ - min_)
    if background is not None:
        rv[background] = img[background]
    return rv
