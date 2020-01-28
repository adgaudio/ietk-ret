"""
TODO: In Progress
"""
import numpy as np
import cv2
import scipy as sp
from dehaze import get_dark_channel
from ietk import methods
from ietk import util


def reshape_A(A, I_shape):
    if np.shape(A) == ():  # scalar
        A = np.reshape(A, (1,1,1))
    elif np.shape(A) == (3,):  # rgb pixel color
        A = np.reshape(A, (1,1,3))
    elif np.shape(A) == (I_shape[0], I_shape[1]):  # gray img
        A = np.reshape(A, I_shape[0], I_shape[1], 1)
    else:  # full channel img
        assert A.shape == I_shape
    return A


def reshape_t(t, I_shape):
    sh = np.shape(t)
    if sh == ():  # scalar
        t = np.reshape(t, (1,1,1))
    else:
        assert t.shape == I_shape[:2]
        t = t.reshape(*t.shape, 1)
    return t


def illuminate_sharpen(
    I, ill_dark_channel_filter_size=50, ill_guided_filter_radius=100,
    ill_guided_eps=1e-8, ill_A=1, sh_t=0.20, sh_blur_radius=75, sh_blur_guided_eps=1e-8):
    """
    Simultaneously Illuminate and Sharpen an image.
    Not equivalent to sharpen(illuminate(img)), since computes J in one pass
    and sharpening computes A=blur(I) rather than A=blur(illuminated(I)).  This
    ends up having quite different results!

    `I` - a [0,1] normalized image of shape (h,w,ch)

    illumination hyperparams:
        `ill_A` - atmosphere, usually (1,1,1) - rgb color (r,g,b) or an array of shape (h,w,ch)
        ill_dark_channel_filter_size
        ill_guided_filter_radius
        ill_guided_eps
    Sharpen hyperparams:
        sh_t - the transmission map for sharpening.  how quickly to amplify differences.
        sh_blur_radius
        sh_blur_guided_eps
    """
    A1 = reshape_A(ill_A, I.shape)
    assert A1.max() <= 1 and A1.min() > 0

    t_unrefined = get_dark_channel(
        (1-I) / ill_A, filter_size=ill_dark_channel_filter_size)
    t1 = 1 - cv2.ximgproc.guidedFilter(
        #  1-I.astype('float32'),  # same
        I.astype('float32'), t_unrefined.astype('float32'),
        ill_guided_filter_radius, ill_guided_eps)
    t1 = t1.clip(.00001, 1)
    t1 = reshape_t(t1, I.shape)
    #  J1 = 1 - ((1-I - ill_A)/t1 + A1)
    #  return J1
    # TODO: sharpen

    # blurring, faster than ndi.gaussian_filter(I)
    #  kernel = np.outer(*([sp.stats.norm.pdf(np.linspace(-1, 1, sh_blur_radius), 0, .7)]*2))
    #  A2 = sp.signal.fftconvolve(I/t1, kernel[:,:,np.newaxis], axes=(0,1))
    A2 = cv2.ximgproc.guidedFilter(
        #  radiance.astype('float32'),
        (I).astype('float32'),
        (I).astype('float32'),
        sh_blur_radius, sh_blur_guided_eps)

    t2 = reshape_t(sh_t, I.shape)
    #  dz = np.array(np.gradient(I))/2+.5
    #  dzn = (dz - dz.min((1,2), keepdims=True)) / dz.ptp((1,2), keepdims=True)
    #  t2 = reshape_t(dzn.mean((0,3)), I.shape)

    #  J2 = (I - A2*(1-t2))/t2
    #  Jb = 1/t2 + (A1-(1-I))/(t1*t2) - A1/t2 - A2/t2 + A2
    #  Jb2 = util.norm01(Jb, bg).clip(0,1)

    #  Jb = 1/t2 + (A1-(1-I))/(t1*t2) - A1/t2 - A2/t2 + A2
    Jb = ((I-(1-A1))/t1 +(1-A1)-A2)/t2 + A2
    Jb2 = util.norm01(Jb, bg).clip(0,1)

    # average with image
    #  Jc = ((util.norm01(Jb, bg) + img)/2).clip(0,1)
    #  Jc /= min(Jc.max(), I.max())
    Jc = Jb2/2 + I/2  # TODO: jb2/2
    Jc2 = util.norm01(Jc, bg)
    # geometric avg with image (better looking)
    #  Jc = np.sqrt(Jb.clip(0, 100)*I)
    #  Jc2 = util.norm01(Jc, bg).clip(0,1)

    best = methods.sharpen(Jc2, focus_region=~bg)
    paper = methods.illuminate_sharpen(I, focus_region=~bg)
    globals().update(locals())
    return best
    # sigmoid average
    #  import matplotlib.colors as C
    #  Jd = C.rgb_to_hsv(I)
    #  z = Jb[:,:,2].copy()
    #  z = ndi.maximum_filter(z, size=3) - ndi.minimum_filter(z, size=3)
    #  z2 = C.rgb_to_hsv(Jb2)[:,:,2]
    #  z2 = ndi.maximum_filter(z2, size=15) - ndi.minimum_filter(z2, size=15)
    #  D = z - z2
    #  #  D = Jb2 - I
    #  b = -D.min() / D.ptp()
    #  D = (D-D.min())/D.ptp()
    #  a = 5
    #  print(b)
    #  _W = 1/(1+np.exp(-1.0*a*(D - b)))
    #  W = (_W - _W.min()) / _W.ptp()
    #  Jd[:,:,2] = (W) * z + (1-W) * z2
    #  Jd = C.hsv_to_rgb(Jd)
    #  #  Jd = ((util.norm01(Jb, bg) + img)/2).clip(0,1)
    #  # sharpen illuminate
    #  #  Jd = 1 - A1 - 1/t1+(I-A2)/(t2*t1) + A2/t1 + A1/t1

    #  plt.figure() ; plt.imshow(Jd)
    #  plt.figure() ; plt.imshow(Jc)

    #  # illuminate sharpen, I and A swapped
    #  Je = 1/t2 + (A1-(1-A2))/(t1*t2) - A1/t2 - I/t2 + I
    #  Je2 = util.norm01(Je, bg).clip(0,1)
    globals().update(locals())
    print(Jb.max(), Jb.min(), Je.max(), Je.min())
    #  return Jb, Jc, Je  # TODO


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    from ietk.data import IDRiD
    from ietk import util

    #  I =  np.dstack([(np.outer(*[np.exp(-np.linspace(0, 1, 1000))]*2))]*3)
    #  bg = np.zeros_like(I, dtype='bool')
    #  z = illuminate_sharpen(I)
    #  import sys ; sys.exit()

    def sh(I, t=.20):  # custom sharpen debugging
        A2 = cv2.ximgproc.guidedFilter(
            #  radiance.astype('float32'),
            (I).astype('float32'),
            (I).astype('float32'),
            sh_blur_radius, sh_blur_guided_eps)
        #  kernel = np.outer(*([sp.stats.norm.pdf(np.linspace(-1, 1, sh_blur_radius), 0, .7)]*2))
        #  A2 = sp.signal.fftconvolve(I/t1, reshape_A(kernel))
        return (I-A2)/t + A2
    dset = IDRiD('./data/IDRiD_segmentation')
    img, labels = dset['IDRiD_25']
    #  he = labels['HE']
    #  ma = labels['MA']
    #  ex = labels['EX']
    #  se = labels['SE']
    #  od = labels['OD']
    # set background pure black.
    bg = util.get_background(img)
    img[bg] = 0

    best = illuminate_sharpen(img)
    illum = methods.illuminate_dcp(I, focus_region=~bg)
    plt.figure(1) ; plt.imshow(sh(illum))
    plt.figure(2) ; plt.imshow(sh(Jc2, .2))
    #  best = competing_methods.sharpen(Jc2, focus_region=~bg)
    import sys ; sys.exit()
    plt.figure() ; plt.imshow(img)
    plt.figure() ; plt.imshow(best)
    plt.figure() ; plt.imshow(util.norm01(best, bg))
    import sys ; sys.exit()
    from sharpen_img import sharpen
    #  shar
    #  sharpen(util.norm01(np.sqrt(Jb.clip(0, 100)*I), bg),
    plt.figure() ; plt.imshow(best2);
    z = (img - Je) / .8 + Je
    plt.imshow(z);
    plt.imshow(Jd)
    plt.savefig('data/test_sharp_ill.png')
    import sys ; sys.exit()
    import mpl_toolkits.axes_grid1
    f = plt.figure(figsize=(20,20))
    axs = mpl_toolkits.axes_grid1.axes_grid.ImageGrid(f, 111, (2, 3))

    axs[0].imshow(img)
    axs[1].imshow(util.norm01(Jb, bg).clip(0,1))
    axs[2].imshow(Jc/Jc.max())
    axs[3].imshow(methods.illuminate_sharpen(img, focus_region=~bg))
    JS = methods.sharpen(Jc/Jc.max(), focus_region=~bg)
    axs[4].imshow(JS)
    axs[5].imshow(util.norm01(methods.illuminate_dcp(img, focus_region=~bg), bg))
    f.savefig('data/test_illuminate_sharpen_atonce.png')


    #  #  J2 = competing_methods.illuminate_dcp(img, focus_region=util.get_foreground(img))
    #  #  print(np.sqrt(np.sum(np.abs(J.clip(0,1) - J2))))

    #  #  z = competing_methods.illuminate_dcp(img, focus_region=~bg)
    #  #  z = competing_methods.sharpen(util.norm01(z), focus_region=~bg)
    #  #  plt.figure(); plt.imshow(util.norm01(z, bg))
    #  #  z = competing_methods.sharpen(img, focus_region=~bg)
    #  #  z = competing_methods.illuminate_dcp(util.norm01(z, bg), focus_region=~bg)
    #  #  plt.figure(); plt.imshow(util.norm01(z))
