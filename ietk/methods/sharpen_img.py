import cv2.ximgproc
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage as ndi

from ietk.data import IDRiD
from ietk import util


def sharpen(img, bg=None, t='laplace', blur_radius=30, blur_guided_eps=1e-8,
            use_guidedfilter=True):
    """Use distortion model to deblur image.  Equivalent to usharp mask:

        1/t * img - (1-1/t) * blurry(img)

    Then, apply guided filter to smooth result but preserve edges.

    img - image to sharpen.
    bg - image background
    t - the transmission map (inverse amount of sharpening)
        can be scalar, matrix of same (h, w) as img, or 3 channel image.
        By default, use a multi-channel sharpened laplace filter on a smoothed
        image with 10x10 kernel. For enhancing fine details in large images.
    """
    if bg is None:
        bg = np.zeros_like(img, dtype='bool')
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
    #  assert np.isnan(A).sum() == 0

    if t == 'laplace':
        t = 1-util.norm01(sharpen(
            ndi.gaussian_laplace(img, (10,10,1)), bg, 0.15), bg)

    if len(np.shape(t)) + 1 == len(img.shape):
        t_refined = np.expand_dims(t, -1).astype('float')
    else:
        t_refined = t
    if np.shape(t):
        t_refined[bg] = 1  # ignore background, but fix division by zero
    J = (  # Eq. 22 of paper
        img.astype('float')-A) / np.maximum(1e-8, np.maximum(t_refined, np.min(t_refined)/2)) + A
    #  assert np.isnan(J).sum() == 0
    if bg is not None:
        J[bg] = 0

    if not use_guidedfilter:
        return J

    r2 = cv2.ximgproc.guidedFilter(
        img.astype('float32'),
        J.astype('float32'),
        2, 1e-8)
    #  assert np.isnan(r2).sum() == 0
    if bg is not None:
        r2[bg] = 0
    return r2


if __name__ == "__main__":
    dset = IDRiD('./data/IDRiD_segmentation')
    #  img, labels = dset['IDRiD_24']
    img_id, img, labels = dset.sample()
    print(img_id)
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

    f, axs = plt.subplots(1, 2, figsize=(15, 5))
    #  f, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img)
    axs[0].set_title('Unmodified image')
    axs[1].imshow(J)
    axs[1].set_title('Sharpened')

    #  axs[2].imshow(J_nogf)
    #  axs[2].set_title('Sharpened without guided filter')
    [ax.axis('off') for ax in axs]
    f.savefig('./paper/figures/sharpen_fundus.png', bbox_inches='tight')
    plt.show()
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


# y = util.norm01(sharpen(ndi.gaussian_laplace((A/2+X/2), (10,10,1)).mean(-1, keepdims=0), bg[:,:,0]), bg[:,:,0])
#  sh(sharpen(Z, bg[:,:,0], (1-y[:,:,np.newaxis])))
