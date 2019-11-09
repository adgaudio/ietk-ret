import cv2.ximgproc
import numpy as np
import util
from idrid import IDRiD
from matplotlib import pyplot as plt


def sharpen(img, bg, t=0.15, blur_radius=30, blur_guided_eps=1e-2):
    """Use distortion model to deblur image.  Equivalent to usharp mask.

        1/t * img - (1-1/t) * blurry(img).

    Then, apply guided filter to smooth result but preserve edges.

    img - image to sharpen.
    bg - image background
    """
    # blurring (faster than ndi.gaussian_filter(I)
    A = cv2.ximgproc.guidedFilter(
        #  radiance.astype('float32'),
        img.astype('float32'),
        img.astype('float32'),
        blur_radius, blur_guided_eps)
    if bg is not None:
        A[bg] = 0

    #  t_refined = np.ones(img.shape[:2]) * t

    if np.shape(t):
        t_refined = np.expand_dims(t, -1).astype('float')
        t_refined[bg] = 1  # division by zero
    else:
        t_refined = t
    J = (  # Eq. 22 of paper
        img.astype('float')-A) / t_refined + A
    if bg is not None:
        J[bg] = 0

    r2 = cv2.ximgproc.guidedFilter(
        img.astype('float32'),
        J.astype('float32'),
        2, 1e-2)
    if bg is not None:
        r2[bg] = 0
    return r2


if __name__ == "__main__":
    dset = IDRiD('./data/IDRiD_segmentation')
    img, labels = dset['IDRiD_24']
    #  he = labels['HE']
    #  ma = labels['MA']
    #  ex = labels['EX']
    #  se = labels['SE']
    #  od = labels['OD']
    # set background pure black.
    bg = util.get_background(img)
    img[bg] = 0


    J = sharpen(img, bg)
    #  r2 = cv2.ximgproc.guidedFilter(
        #  img.astype('float32'),
        #  J.astype('float32'),
        #  2, 1e-2)
    #  r2[bg] = 0


    plt.figure(num=1) ; plt.imshow(img)
    plt.figure(num=2) ; plt.imshow(util.norm01(J, None))
    plt.figure(num=3) ; plt.imshow(J.clip(0, 1))
    #  plt.figure(num=4) ; plt.imshow(util.norm01(r2, bg))
    #  plt.figure(num=5) ; plt.imshow(r2.clip(0, 1))

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
