"""
sharpen an image.  in progress.
"""
import scipy.ndimage as ndi
import dehaze as D
from dehaze import *
import util
from idrid import IDRiD
from matplotlib import pyplot as plt

def mm(im):
    rv = (im.min(), im.max())
    print(rv)
    return rv

dark_channel_filter_size=15
guided_filter_radius=20
guided_eps=1e-2

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

#  A = np.dstack([ndi.gaussian_filter(img[:, :, ch], sigma=20)
               #  for ch in range(img.shape[2])])
# faster blurring
A = cv2.ximgproc.guidedFilter(
    #  radiance.astype('float32'),
    img.astype('float32'),
    img.astype('float32'),
    guided_filter_radius, guided_eps)
A[bg] = 0


#  t_unrefined = 1 - get_dark_channel(img, dark_channel_filter_size)
    #  t_unrefined = np.maximum(t_unrefined, 0.4)

    # refine dark channel using guided filter (ie like a bilateral filter or
    # anisotropic diffusion but faster)
    #  t_refined = cv2.ximgproc.guidedFilter(
    #      img.astype('float32'),
    #      t_unrefined.astype('float32'), guided_filter_radius, guided_eps)
    #  t_refined = t_refined.clip(0.0001, 1)  # guided filter can make slightly >1
#  print(t_unrefined.min(), t_unrefined.max())
#  t_refined = t_unrefined.clip(0.0001, 1)  # guided filter can make slightly >1
t_refined = np.ones(img.shape[:2]) * .2
t_refined = np.ones(img.shape[:2]) * .1


radiance = (  # Eq. 22 of paper
    img.astype('float')-A) \
    / np.expand_dims(t_refined, -1).astype('float') \
    + A
    #  radiance = norm01(radiance)
    #  radiance = radiance / radiance.max()
#  radiance = radiance.clip(0, 1)

r2 = cv2.ximgproc.guidedFilter(
    img.astype('float32'),
    radiance.astype('float32'),
    2, guided_eps)
r2[bg] = 0


plt.figure(num=1) ; plt.imshow(img)
plt.figure(num=2) ; plt.imshow(util.norm01(radiance, None))
plt.figure(num=3) ; plt.imshow(radiance.clip(0, 1))
plt.figure(num=4) ; plt.imshow(util.norm01(r2, bg))
plt.figure(num=5) ; plt.imshow(r2.clip(0, 1))

import evaluation as EV
import metric

EV.empirical_evaluation(
    {'img': img,
     'radiance': radiance,
     'r2': r2,
     'norm01': util.norm01(radiance, bg),
     'clip': radiance.clip(0, 1),
     'r2norm01': util.norm01(r2, bg),
     'r2clip': r2.clip(0, 1),
     },
    labels, metric.eval_methods, bg, num_procs=8)
