import numpy as np
import scipy.ndimage as ndi
from scipy import signal
import cv2
import logging
log = logging.getLogger(__name__)



def get_background(img, is_01_normalized=True):
    return ~get_foreground(img, is_01_normalized)



def get_foreground(img, is_01_normalized=True):
    return center_crop_and_get_foreground_mask(
        img, crop=False, is_01_normalized=is_01_normalized)[1]


def get_center_circle_coords(im, is_01_normalized: bool):
    A = np.dstack([
        signal.cspline2d(im[:,:,ch] * (255 if is_01_normalized else 1), 200.0)
        for ch in range(im.shape[-1])])
    min_r = int(min(im.shape[0], im.shape[1]) / 4)
    max_r = int(max(im.shape[0], im.shape[1]) / 4*3)
    try:
        circles = cv2.HoughCircles(
            (norm01(A).max(-1)*255).astype('uint8'), cv2.HOUGH_GRADIENT, .8,
            min(A.shape[:2]), param1=20, param2=50, minRadius=min_r, maxRadius=max_r)[0]
    except:
        log.warn('center_crop_and_get_foreground_mask failed to get background - trying again with looser parameters')
        A2 = get_foreground_slow(im)
        circles = cv2.HoughCircles(
            (A2*255).astype('uint8'), cv2.HOUGH_GRADIENT, .8,
            min(A.shape[:2]), param1=20, param2=10, minRadius=min_r, maxRadius=max_r)[0]
    x, y, r = circles[circles[:, -1].argmax()].round().astype('int')
    return x,y,r



def get_foreground_mask_from_center_circle_coords(shape, x,y,r):
    mask = np.zeros(shape, dtype='uint8')
    cv2.circle(mask, (x, y), r, 255, cv2.FILLED)
    mask = mask.astype(bool)
    return mask


def center_crop_and_get_foreground_mask(im, crop=True, is_01_normalized=True, center_circle_coords=None, label_img=None):
    if center_circle_coords is not None:
        x,y,r = center_circle_coords
    else:
        h, w, _ = im.shape
        x, y, r = get_center_circle_coords(im, is_01_normalized)
    mask = get_foreground_mask_from_center_circle_coords(im.shape[:2], x,y,r)
    if crop:
        crop_slice = np.s_[max(0, y-r):min(h,y+r),max(0,x-r):min(w,x+r)]
        rv = [im[crop_slice], mask[crop_slice]]
        if label_img is not None:
            rv.append(label_img[crop_slice])
    else:  # don't crop.  just get the mask.
        rv = [im, mask]
        if label_img is not None:
            rv.append(label_img[crop_slice])
    return rv


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
