import scipy.ndimage as ndi
import numpy as np
from cv2.ximgproc import guidedFilter
from matplotlib import pyplot as plt

from ietk.methods.sharpen_img import sharpen
from ietk import util
from ietk.data import IDRiD


def solvet(I, A, use_gf=True, fsize=(5,5)):
    z = 1-ndi.minimum_filter((I/A).min(-1), fsize)
    if use_gf:
        z = gf(I, z)
    rv = z.reshape(*I.shape[:2], 1)
    return rv

#  def solvet_perchannel(I, A, use_gf=True, fsize=(5,5,0)):
#      z = 1-ndi.minimum_filter((I/A), fsize)
#      if use_gf:
#          z = gf(I, z)
#      return z

def solvetmax(I, A):
    z = 1-ndi.maximum_filter((I/A).max(-1), (5, 5))
    return gf(I, z).reshape(*I.shape[:2], 1)

def solveJ(I, A, t):
    epsilon = max(np.min(t)/2, 1e-8)
    return (I-A)/np.maximum(t, epsilon) + A

def gf(guide, src, r=100, eps=1e-8):
    return guidedFilter(guide.astype('float32'), src.astype('float32'), r, eps).astype('float64')

def ta(img, ignore_ch=None, **kws):
    if ignore_ch is not None:
        I = img.copy()
        I[:,:,ignore_ch] = 0
    else:
        I = img
    return solvet(1-img, 1, **kws)

def td(img, ignore_ch=None, **kws):
    if ignore_ch is not None:
        I = img.copy()
        I[:,:,ignore_ch] = 0
    else:
        I = img
    return 1-solvet(1-img, 1, **kws)

def tb(img, ignore_ch=None, **kws):
    if ignore_ch is not None:
        I = img.copy()
        I[:,:,ignore_ch] = 1
    else:
        I = img
    return solvet(I, 1, **kws)

def tc(img, ignore_ch=None, **kws):
    if ignore_ch is not None:
        I = img.copy()
        I[:,:,ignore_ch] = 1
    else:
        I = img
    return 1-solvet(I, 1, **kws)

def A(img):
    return solveJ(img, 0, ta(img))
def B(img):
    return solveJ(img, 0, tb(img))
def C(img):
    return solveJ(img, 0, tc(img))
def D(img):
    return solveJ(img, 0, td(img))
def W(img):
    return solveJ(img, 1, ta(img))
def X(img):
    return solveJ(img, 1, tb(img))
def Y(img):
    return solveJ(img, 1, tc(img))
def Z(img):
    return solveJ(img, 1, td(img))
def B_ret(img):
    """specific to retinal fundus images, where blue channel is too sparse"""
    return solveJ(img, 0, tb(img, ignore_ch=2))
def C_ret(img):
    """specific to retinal fundus images, where blue channel is too sparse"""
    return solveJ(img, 0, tc(img, ignore_ch=2))
def X_ret(img):
    """specific to retinal fundus images, where blue channel is too sparse"""
    return solveJ(img, 1, tb(img, ignore_ch=2))
def Y_ret(img):
    """specific to retinal fundus images, where blue channel is too sparse"""
    return solveJ(img, 1, tc(img, ignore_ch=2))


def brighten_darken(img, method_name: str, focus_region=None,
                    fundus_image: bool=True):
    """
    Apply a brightening or darkening method, following the ICIAR2020 paper,
    Enhancement of Retinal Fundus Images via Pixel Color Amplification.

    `img`: a (h,w,c) fundus image normalized into [0,1] range.  The
      number of color channels, c > 0.

    `method_name: str` - any combination of letters individual letters
    'ABCDWXYZ', each optionally prefixed by 's' and separated by a '+'. The
    's' performs sharpening.  Each of the other letters refers to a brightening
    (A,B,C or D) or darkening (W,X,Y,Z) operation.  See the Pixel Color
    Amplification paper for details (ICIAR 2020).
    Example method names you can try: 'A+B+W+X' or 'sA+sC+sX+sZ'

    `focus_region` - a foreground boolean mask specifying which pixels of image
    to sharpen

    `fundus_image: bool` By default, assume retinal fundus images, where
    B,C,X,Y all ignore the blue channel.  If you want to brighten or darken
    different image domains, you're probably going to want to optimize the
    neighborhood size in solvet(fsize=(...)) and the guided filter parameters
    gf(...).  In this case, you should just build your own function using
    solvet and solveJ directly.
    """
    func_names = method_name.split('+')
    if fundus_image:
        _methods = dict(zip('ABCDWXYZ', [A,B_ret,C_ret,D,W,X_ret,Y_ret,Z]))
    else:
        _methods = dict(zip('ABCDWXYZ', [A,B,C,D,W,X,Y,Z]))
    I2 = np.zeros_like(img)
    for func_name in func_names:
        tmp = _methods[func_name.lstrip('s')](img)
        if func_name.startswith('s'):
            tmp = sharpen(tmp, ~focus_region)
        I2 += tmp
    I2 /= len(func_names)
    return I2


def resizeforplot(img):
    import PIL
    size = (np.array(img.shape)/2).astype('int')
    return np.asarray(PIL.Image.fromarray((img.clip(0, 1)*255).astype('uint8')).resize(
        (size[1], size[0])))


if __name__ == "__main__":
    import os
    os.makedirs('data/plots/brighten_darken/', exist_ok=True)  # save dir
    # load an image
    dset = IDRiD('./data/IDRiD_segmentation')
    img_id, img, labels = dset.sample()
    print("using image", img_id)
    #  img, labels = dset['IDRiD_46']
    #  he = labels['HE']
    #  ma = labels['MA']
    #  ex = labels['EX']
    #  se = labels['SE']
    #  od = labels['OD']
    # set background pure black.

    I = img.copy()
    #  I = img[1500:2500, 1500:2500, :]
    #  labels = {k: v[1500:2500, 1500:2500] for k, v in labels.items()}

    #  bg = np.zeros_like(I, dtype='bool')
    I, fg = util.center_crop_and_get_foreground_mask(I)
    bg = ~fg
    I[bg] = 0

    # four transmission maps, for retinal images
    a,b,c,d = [ta(I), tb(I, ignore_ch=2), tc(I, ignore_ch=2), td(I)]

    #  a = solvet(1-I, 1)  # == 1-solvetmax(I, 1)
    #  d = 1-solvet(1-I, 1)  # == solvetmax(I, 1)
    #  I2 = I.copy()
    #  I2[:,:,2] = 1  # the min values of blue channel is too noise
    #  c = 1-solvet(I2, 1)  # == solvetmax(1-I, 1)
    #  b = solvet(I2, 1)  # == 1-solvetmax(1-I, 1)

    #  #  a = solvet(1-I, 1, False, (50,20))  # == 1-solvetmax(I, 1)
    #  #  d = 1-solvet(1-I, 1, False, (50,20))  # == solvetmax(I, 1)
    #  #  c = 1-solvet(I, 1, False, (50,20))  # == solvetmax(1-I, 1)
    #  #  b = solvet(I, 1, False, (50,20))  # == 1-solvetmax(1-I, 1)

    # Brighten
    A, B, C, D = A(I), B_ret(I), C_ret(I), D(I)
    # Darken
    W, X, Y, Z = W(I), X_ret(I), Y_ret(I), Z(I)

    # Plots
    print('plotting')

    kws = dict(bt='\mathbf{t}', bI='\mathbf{I}', bA='\mathbf{A}')
    t_eqs = [
        r'${bt} = $solve_t$(1-{bI}, {bA}=1)$'.format(**kws),
        r'${bt} = $solve_t$({bI}, {bA}=1)$'.format(**kws),
        r'${bt} = 1-$solve_t$({bI}, {bA}=1)$'.format(**kws),
        r'${bt} = 1-$solve_t$(1-{bI}, {bA}=1)$'.format(**kws),
    ]

    f, axs = plt.subplots(2,2, num=1, figsize=(10,10))
    f.suptitle('Transmission maps', fontsize=28)
    axs2 = []
    for n,(ax, t) in enumerate(zip(axs.ravel(), [a,b,c,d])):
        ax.imshow(resizeforplot(t.squeeze()), cmap='gray', interpolation='none')
        #  ax.axis('off')
        #  ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        #  tmpax = ax.twiny()
        #  tmpax.xaxis.set_ticks([])
        #  tmpax.xaxis.set_label_position('bottom')
        #  axs2.append(tmpax)
    axs[0,0].set_ylabel('Weak amplification', fontsize=20)
    axs[1,0].set_ylabel('Strong amplification', fontsize=20)
    axs[1,0].set_xlabel('Amplifying dark regions', fontsize=20)
    axs[1,1].set_xlabel('Amplifying bright regions', fontsize=20)
    #  axs2[0].set_xlabel(t_eqs[0], fontsize=18)
    #  axs2[1].set_xlabel(t_eqs[1], fontsize=18)
    #  axs2[2].set_xlabel(t_eqs[2], fontsize=18)
    #  axs2[3].set_xlabel(t_eqs[3], fontsize=18)
    axs.ravel()[0].set_title(t_eqs[0], fontsize=20)
    axs.ravel()[1].set_title(t_eqs[1], fontsize=20)
    axs.ravel()[2].set_title(t_eqs[2], fontsize=20)
    axs.ravel()[3].set_title(t_eqs[3], fontsize=20)
    f.tight_layout()
    f.subplots_adjust(wspace=0.02, top=0.92)
    f.savefig('./data/plots/brighten_darken/amplifier_t.png', bbox_inches='tight')

    f2, ax = plt.subplots(num=2, figsize=(10,10))
    f2.suptitle('Source Image', fontsize=28)
    #  ax.imshow(resizeforplot(sharpen(I, bg)), interpolation='none')
    ax.imshow(resizeforplot(I), interpolation='none')
    ax.axis('off')
    f2.tight_layout()
    f2.subplots_adjust(wspace=0.02, hspace=0.02, top=0.92)
    f2.savefig('./data/plots/brighten_darken/amplifier_I.png', bbox_inches='tight')

    f3, axs = plt.subplots(2,2, num=3, figsize=(10,10))
    f3.suptitle(r'Whole Image Brightening:  $\mathbf{J} = \frac{\mathbf{I}-\mathbf{0}}{\mathbf{t}} + \mathbf{0}$', fontsize=28)
    f3b, axs3b = plt.subplots(2,2, num=33, figsize=(10,10))
    f3b.suptitle(r'Whole Image Brightening followed by Sharpening', fontsize=28)
    letter1 = ['A.', 'B.', 'C.', 'D.']
    for n,(ax, ax3b, t) in enumerate(zip(axs.ravel(), axs3b.ravel(), [a,b,c,d])):
        J = solveJ(I, 0, t)
        J3b = sharpen(J, bg)
        ax.imshow(resizeforplot(J), interpolation='none')
        ax.set_title(f'{letter1[n]}  {t_eqs[n]}', fontsize=22)
        ax.axis('off')
        ax3b.axis('off')
        ax3b.imshow(resizeforplot(J3b))
        ax3b.set_title(f's{letter1[n]}', fontsize=22)
    f3.tight_layout()
    f3.subplots_adjust(wspace=0.02, hspace=0.02, top=.9)
    f3.savefig('data/plots/brighten_darken/amplifier_b.png', bbox_inches='tight')
    f3b.tight_layout()
    f3b.subplots_adjust(wspace=0.02, hspace=0.02, top=.9)
    f3b.savefig('data/plots/brighten_darken/amplifier_b_sharpen.png', bbox_inches='tight')

    f4, axs = plt.subplots(2,2, num=4, figsize=(10,10))
    f4.suptitle(r'Whole Image Darkening:  $\mathbf{J} = \frac{\mathbf{I}-\mathbf{1}}{\mathbf{t}} + \mathbf{1}$', fontsize=28)
    f4b, axs4b = plt.subplots(2,2, num=44, figsize=(10,10))
    f4b.suptitle(r'Whole Image Darkening followed by Sharpening', fontsize=28)
    letter2 = ['W.', 'X.', 'Y.', 'Z.']
    for n,(ax, ax4b, t) in enumerate(zip(axs.ravel(), axs4b.ravel(), [a,b,c,d])):
        J = solveJ(I, 1, t)
        J4b = sharpen(J, bg)
        ax.imshow(resizeforplot(J), interpolation='none')
        ax.set_title(f'{letter2[n]}  {t_eqs[n]}', fontsize=22)
        ax.axis('off')
        ax4b.axis('off')
        ax4b.imshow(resizeforplot(J4b))
        ax4b.set_title(f's{letter2[n]}', fontsize=22)
    f4.tight_layout()
    f4.subplots_adjust(wspace=0.02, hspace=0.02, top=.9)
    f4.savefig('data/plots/brighten_darken/amplifier_d.png', bbox_inches='tight')
    f4b.tight_layout()
    f4b.subplots_adjust(wspace=0.02, hspace=0.02, top=.9)
    f4b.savefig('data/plots/brighten_darken/amplifier_d_sharpen.png', bbox_inches='tight')
    plt.show(block=False)

    # Extra visuals showing effects of composing images together.  compute
    # intensive.

    #  # Brighten and darken under parallel composition by avg.
    #  from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
    #  from itertools import product
    #  f5 = plt.figure(figsize=(10,10), num=5)
    #  grid = ImageGrid(f5, 111, (4, 4))
    #  for i, (bright, dark) in enumerate(product([A,B,C,D], [W,X,Y,Z])):
    #      grid[i].imshow(resizeforplot(sharpen(bright/2+dark/2, bg)))
    #      #  grid[i].imshow(resizeforplot(bright/2+dark/2))
    #  f5.savefig('data/plots/brighten_darken/compose_parallel_avg.png', bbox_inches='tight')

    #  #  # composition AZ and ZA
    #  bd = (solveJ(i, a, t) for i in [A,B,C,D] for a in [1] for t in [a,b,c,d])
    #  db = (solveJ(i, a, t) for i in [W,X,Y,Z] for a in [0] for t in [a,b,c,d])
    #  from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
    #  f6 = plt.figure(figsize=(10,10), num=6)
    #  f7 = plt.figure(figsize=(10,10), num=7)
    #  grid6 = ImageGrid(f6, 111, (4, 4))
    #  grid7 = ImageGrid(f7, 111, (4, 4))
    #  for i, (b, d) in enumerate(zip(bd, db)):
    #      #  grid6[i].imshow(resizeforplot(sharpen(b, bg)))
    #      #  grid7[i].imshow(resizeforplot(sharpen(d, bg)))
    #      grid6[i].imshow(resizeforplot(b))
    #      grid7[i].imshow(resizeforplot(d))
    #  f6.savefig('data/plots/brighten_darken/compose_series_bd.png', bbox_inches='tight')
    #  f7.savefig('data/plots/brighten_darken/compose_series_db.png', bbox_inches='tight')


    plt.show(block=False)
