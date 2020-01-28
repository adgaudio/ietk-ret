#!/usr/bin/env python
import scipy.ndimage as ndi
import scipy.stats as stats
import numpy as np
from matplotlib import pyplot as plt
import PIL
import glob
import cv2

from ietk.util import get_background


def get_dark_channel(
        img: np.ndarray, filter_size: int):
    """Compute the dark channel of given image.
    This is the pixel-wise min along rgb channels for
    a neighborhood of fixed size around the pixel.  I use a circular footprint
    rather than a rectangle.

    reference: http://kaiminghe.com/publications/pami10dehaze.pdf

    img: np.ndarray of size (h, x, 3)
    filter_size: integer
    """
    _tmp = stats.norm.pdf(np.linspace(0, 1, filter_size), .5, .25/2)
    dark_channel = ndi.minimum_filter(
        img.min(-1), footprint=np.log(np.outer(_tmp, _tmp)) > -6)
    return dark_channel


def get_atmosphere(img: np.ndarray, dark: np.ndarray):
    """Given an image of shape (h, w, 3) and a dark channel of shape (h, w),
    compute and return the atmosphere, a vector of shape (3, )

    Consider the 10% brightest pixels in dark channel, look up their
    intensities in original image and use the brightest intensity found from
    that set.
    """
    # top 10\% of brightest pixels in dark channel
    q = np.quantile(dark.ravel(), 0.999) - 1e-6
    mask = dark >= q
    rv = np.array([img[:, :, ch][mask].max() for ch in range(3)])
    assert img.shape[2] == 3  # sanity check
    rv += 1 - rv.max()  # seems to make img brighter
    return rv


def dehaze(img, dark_channel_filter_size=15, guided_filter_radius=50,
           guided_eps=1e-2):
    img = img / img.max()
    darkch_unnorm = get_dark_channel(img, dark_channel_filter_size)
    A = get_atmosphere(img, darkch_unnorm).reshape(1, 1, 3)
    #  atmosphere_upper_thresh = 220
    #  A = np.maximum(A, atmosphere_upper_thresh)

    t_unrefined = 1 - get_dark_channel(img / A, dark_channel_filter_size)
    #  t_unrefined = np.maximum(t_unrefined, 0.4)

    # refine dark channel using guided filter (ie like a bilateral filter or
    # anisotropic diffusion but faster)
    t_refined = cv2.ximgproc.guidedFilter(
        img.astype('float32'),
        t_unrefined.astype('float32'), guided_filter_radius, guided_eps)
    t_refined = t_refined.clip(0.0001, 1)  # guided filter can make slightly >1

    radiance = (  # Eq. 22 of paper
        img.astype('float')-A) \
        / np.expand_dims(t_refined, -1).astype('float') \
        + A
    #  radiance = norm01(radiance)
    #  radiance = radiance / radiance.max()
    radiance = radiance.clip(0, 1)
    return locals()


def illumination_correction(img, dark_channel_filter_size=25,
                            guided_filter_radius=80, guided_eps=1e-2, A=1):
    """Illumination correction is basically just inverted dehazing"""
    img = img / img.max()

    # notice there is no "1 - get_dark..." in the equation here.
    t_unrefined = get_dark_channel((1-img) / A, dark_channel_filter_size)
    # invert image after guided filtering
    t_refined = 1-cv2.ximgproc.guidedFilter(
        1-img.astype('float32'),
        t_unrefined.astype('float32'), guided_filter_radius, guided_eps)
    t_refined = t_refined.clip(0.00001, 1)  # guided filter can make slightly >1
    # invert the inverted image when recovering radiance
    radiance = 1 - (((1-img.astype('float')) - A)/np.expand_dims(t_refined, -1) + A)
    #  radiance = norm01(radiance)
    #  radiance = radiance / radiance.max()
    #  radiance = radiance.clip(0, 1)
    return locals()


def dehaze_from_fp(fp):
    with PIL.Image.open(fp) as img:
        img.load()
    img = np.array(img)/255
    # remove background, assuming retinal fundus image
    background = get_background(img)
    img[background] = 1
    return dehaze(img)


def illuminate_from_fp(fp):
    with PIL.Image.open(fp) as img:
        img.load()
    img = np.array(img)/255
    return illuminate_dehaze(img)


def illuminate_dehaze(img):
    """
    Perform illumination correction to remove shadows followed by dehazing.
    Correctly remove background
    Return a tuple of dicts.  The first dict is output of illumination
    correction.  Second dict is output from dehazing.
    """
    # compute a background mask to clean up noise from the guided filter
    background = get_background(img)
    img[background] = 1

    d = illumination_correction(img)
    # reset the background
    d['radiance'][background] = 1/255

    d2 = dehaze(d['radiance'])

    d['background'] = background
    return d, d2


if __name__ == "__main__":
    #  fp = '../../data/tiananmen1.png'
    #  img = plt.imread(fp)
    #  dct = dehaze(img)
    #  img2 = dct['radiance']
    #  print(img2.min(), img2.max())
    #  img2 = (img2 - img2.min()) / (img2.max() - img2.min())
    #  plt.imshow(np.clip(img2, 0, 1))
    #  plt.imshow(np.clip(img2, 0, 1))
    #  import sys ; sys.exit()

    fps_healthy = glob.glob('./data/messidor_grade1/*/*')
    fps_grade3 = glob.glob('./data/messidor_grade3/*/*')

    #  for fp in fps_healthy[:10]:
        #  illuminate_from_fp(fp)
    #  for fp in fps_grade3[:10]:
        #  illuminate_from_fp(fp)

    # testing: check that the methods work
    fp = fps_grade3[0]
    #  #  fp = '../../data/tiananmen1.png'
    #  #  fp = '../../data/forest.jpg'
    with PIL.Image.open(fp) as img:
        img.load()
    img = np.array(img)/255
    d, d2 = illuminate_from_fp(fp)
    f, axs = plt.subplots(1, 3)
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[1].imshow(d['t_refined'], cmap='Greys')
    axs[1].set_title("Illumination Depth Map")
    axs[2].imshow(d2['t_refined'], cmap='Greys')
    axs[2].set_title("Dehaze Depth Map")
    f.suptitle('Depth maps for given input image')

    illuminated = d['radiance']
    dehazed = d2['radiance']
    f, axs = plt.subplots(1, 3)
    axs[0].imshow(img)
    axs[0].set_title('Original Image')
    axs[1].imshow(illuminated)
    axs[1].set_title("Illuminated")
    axs[2].imshow(dehazed)
    axs[2].set_title("Dehazed")
    f.suptitle('Illumination Correction Pipeline')


    # dehaze
    d = dehaze_from_fp(fp)
    f, axs = plt.subplots(1, 3)
    axs[0].imshow(img)
    axs[0].set_title("Original Image")
    axs[1].imshow(d['t_refined'], cmap='Greys')
    axs[1].set_title("Dehaze Depth Map")
    axs[2].imshow(d['radiance'])
    axs[2].set_title("Dehazed")
    f.suptitle('Dehazing the image')
    #  import sys ; sys.exit()
    plt.show()
