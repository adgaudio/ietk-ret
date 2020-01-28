import numpy as np
import scipy.ndimage as ndi
from scipy import signal
import cv2
import logging
log = logging.getLogger(__name__)



def get_background(img, is_01_normalized=True):
    return ~get_foreground(img, is_01_normalized)



def get_foreground(img, is_01_normalized=True):
    return center_crop_and_get_foreground_mask(img, False, is_01_normalized)[1]


def center_crop_and_get_foreground_mask(im, crop=True, is_01_normalized=True):
    A = np.dstack([
        signal.cspline2d(im[:,:,ch] * (255 if is_01_normalized else 1), 200.0)
        for ch in range(im.shape[-1])])
    try:
        circles = cv2.HoughCircles(
            (norm01(A).max(-1)*255).astype('uint8'), cv2.HOUGH_GRADIENT, .8,
            min(A.shape[:2]), param1=20, param2=50, minRadius=400, maxRadius=2500)[0]
    except:
        log.warn('center_crop_and_get_foreground_mask failed to get background - trying again with looser parameters')
        A2 = get_foreground_slow(im)
        circles = cv2.HoughCircles(
            (A2*255).astype('uint8'), cv2.HOUGH_GRADIENT, .8,
            min(A.shape[:2]), param1=20, param2=10, minRadius=400, maxRadius=2500)[0]
    h, w, _ = im.shape
    x, y, r = circles[circles[:, -1].argmax()].round().astype('int')
    mask = np.zeros(im.shape[:2], dtype='uint8')
    cv2.circle(mask, (x, y), r, 255, cv2.FILLED)
    mask = mask.astype(bool)
    if crop:
        crop_slice = np.s_[max(0, y-r):min(h,y+r),max(0,x-r):min(w,x+r)]
        return im[crop_slice], mask[crop_slice]
    else:
        return im, mask


def get_background_slow(img):
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
    return background.sum(2) == 3


def get_foreground_slow(img):
    return ~get_background_slow(img)


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
