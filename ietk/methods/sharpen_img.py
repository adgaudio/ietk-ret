import cv2.ximgproc
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
import logging

from ietk.data import IDRiD
from ietk import util

log = logging.getLogger(__name__)


def check_and_fix_nan(A, replacement_img):
    nanmask = np.isnan(A)
    if nanmask.any():
        log.warn("sharpen: guided filter blurring operation or laplace filter returned nans. your input image has extreme values")
        A[nanmask] = replacement_img[nanmask]
    return A


def sharpen(img, bg=None, t='laplace', blur_radius=30, blur_guided_eps=1e-8,
            use_guidedfilter='if_large_img'):
    """Use distortion model to deblur image.  Equivalent to usharp mask:

        1/t * img - (1-1/t) * blurry(img)

    Then, apply guided filter to smooth result but preserve edges.

    img - image to sharpen, assume normalized in [0,1]
    bg - image background
    t - the transmission map (inverse amount of sharpening)
        can be scalar, matrix of same (h, w) as img, or 3 channel image.
        By default, use a multi-channel sharpened laplace filter on a smoothed
        image with 10x10 kernel. For enhancing fine details in large images.
    use_guidedfilter - a bool or the string 'if_large_img' determining whether
        to clean up the resulting sharpened image.  If the min image dimension is
        less that 1500, this cleanup operation may blur the
        image, ruining its quality.
    """
    if bg is None:
        bg = np.zeros(img.shape[:2], dtype='bool')
    else:
        img = img.copy()
        img[bg] = 0
    #  assert np.isnan(img).sum() == 0
    #  assert np.isnan(t).sum() == 0

    # blurring (faster than ndi.gaussian_filter(I)
    A = cv2.ximgproc.guidedFilter(
        #  radiance.astype('float32'),
        img.astype('float32'),
        img.astype('float32'),
        blur_radius, blur_guided_eps)

    if t == 'laplace':
        t = 1-util.norm01(sharpen(ndi.morphological_laplace(
            img, (2,2,1), mode='wrap'), bg, 0.15), bg)
        #  t = 1-util.norm01(ndi.morphological_laplace(
            #  img, (2,2,1), mode='wrap'), bg)

        # todo note: laplace t is 01 normalized.  should we keep the max
        # and just normalize the lower range (or vice versa or something)?

        # note2: if laplace is all zeros (due to bad input img), t will be all nan.


    if len(np.shape(t)) + 1 == len(img.shape):
        t_refined = np.expand_dims(t, -1).astype('float')
    else:
        t_refined = t
    if np.shape(t):
        t_refined[bg] = 1  # ignore background, but fix division by zero
    J = (
        img.astype('float')-A) / np.maximum(1e-8, np.maximum(t_refined, np.min(t_refined)/2)) + A
    #  assert np.isnan(J).sum() == 0
    if bg is not None:
        J[bg] = 0

    # applying a guided filter for smoothing image at this point can be
    # problematic to the image quality, significantly blurring it.
    if use_guidedfilter == 'if_large_img':
        # note: at some point, find a better threshold?  This works.
        use_guidedfilter = min(J.shape[0], J.shape[1]) >= 1500
    if not use_guidedfilter:
        J = check_and_fix_nan(J, img)
        return J

    r2 = cv2.ximgproc.guidedFilter(
        img.astype('float32'),
        J.astype('float32'),
        2, 1e-8)
    r2 = check_and_fix_nan(r2, img)
    if bg is not None:
        r2[bg] = 0
    return r2


if __name__ == "__main__":
    import os
    os.makedirs('./data/plots', exist_ok=True)
    dset = IDRiD('./data/IDRiD_segmentation')
    img, labels = dset['IDRiD_26']
    #  img_id, img, labels = dset.sample()
    #  print(img_id)
    #  he = labels['HE']
    #  ma = labels['MA']
    #  ex = labels['EX']
    #  se = labels['SE']
    #  od = labels['OD']
    # set background pure black.
    bg = util.get_background(img)
    img[bg] = 0

    J = sharpen(img, bg, .15)
    J_nogf = sharpen(img, bg, .15, use_guidedfilter=False)
    J_laplace = sharpen(img, bg)

    #  f, axs = plt.subplots(1, 2, figsize=(15, 5))
    f, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title('Unmodified image', fontsize=22)
    axs[1].imshow(J)
    axs[1].set_title('Sharpened, Algo. 1', fontsize=22)
    #  axs[2].imshow(J_nogf)
    #  axs[2].set_title('Sharpened without guided filter')
    axs[2].imshow(J_laplace)
    axs[2].set_title('Sharpened, Algo. 2', fontsize=22)
    #  axs[2].imshow(J_nogf)
    #  axs[2].set_title('Sharpened without guided filter')
    [ax.axis('off') for ax in axs]
    f.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0.1)
    f.savefig('./data/plots/sharpen_fundus.png', bbox_inches='tight')

    #  plt.figure(num=4) ; plt.imshow(util.norm01(r2, bg))
    #  plt.figure(num=5) ; plt.imshow(r2.clip(0, 1))

    plt.show()
    #  import evaluation as EV
    #  import metric

    #  EV.empirical_evaluation(
        #  {'img': img,
        #  'radiance': J,
        #  'r2': r2,
        #  'norm01': util.norm01(J, bg),
        #  'clip': J.clip(0, 1),
        #  'r2norm01': util.norm01(r2, bg),
        #  'r2clip': r2.clip(0, 1),
        #  },
        #  labels, metric.eval_methods, bg, num_procs=8)


# y = util.norm01(sharpen(ndi.gaussian_laplace((A/2+X/2), (10,10,1)).mean(-1, keepdims=0), bg[:,:,0]), bg[:,:,0])
#  sh(sharpen(Z, bg[:,:,0], (1-y[:,:,np.newaxis])))
