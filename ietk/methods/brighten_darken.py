import scipy.ndimage as ndi
import numpy as np
from cv2.ximgproc import guidedFilter
from matplotlib import pyplot as plt
import PIL

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


def resizeforplot(img):
    import PIL
    size = (np.array(img.shape)/2).astype('int')
    return np.asarray(PIL.Image.fromarray((img.clip(0, 1)*255).astype('uint8')).resize(
        (size[1], size[0])))

if __name__ == "__main__":
    from ietk import methods
    from ietk import metric
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
    bg = util.get_background(I)
    I[bg] = 0

    # four transmission maps
    a = solvet(1-I, 1)  # == 1-solvetmax(I, 1)
    d = 1-solvet(1-I, 1)  # == solvetmax(I, 1)
    I2 = I.copy()
    I2[:,:,2] = 1  # the min values of blue channel is too noise
    c = 1-solvet(I2, 1)  # == solvetmax(1-I, 1)
    b = solvet(I2, 1)  # == 1-solvetmax(1-I, 1)

    #  a = solvet(1-I, 1, False, (50,20))  # == 1-solvetmax(I, 1)
    #  d = 1-solvet(1-I, 1, False, (50,20))  # == solvetmax(I, 1)
    #  c = 1-solvet(I, 1, False, (50,20))  # == solvetmax(1-I, 1)
    #  b = solvet(I, 1, False, (50,20))  # == 1-solvetmax(1-I, 1)

    # Brighten
    A, B, C, D = [solveJ(I, 0, t) for t in [a,b,c,d]]
    # Darken
    W, X, Y, Z = [solveJ(I, 1, t) for t in [a,b,c,d]]

    # Plots
    print('plotting')

    kws = dict(bt='\mathbf{t}', bI='\mathbf{I}', bA='\mathbf{A}')
    t_eqs = [
        r'${bt} = $solve_t$(1-{bI}, {bA}=1)$'.format(**kws),
        r'${bt} = $solve_t$({bI}, {bA}=1)$'.format(**kws),
        r'${bt} = 1-$solve_t$({bI}, {bA}=1)$'.format(**kws),
        r'${bt} = 1-$solve_t$(1-{bI}, {bA}=1)$'.format(**kws),
    ]

    f, axs = plt.subplots(2,2, num=1)
    f.suptitle('Transmission maps')
    axs2 = []
    for n,(ax, t) in enumerate(zip(axs.ravel(), [a,b,c,d])):
        ax.imshow(resizeforplot(t.squeeze()), cmap='gray', interpolation='none')
        #  ax.axis('off')
        ax.xaxis.set_label_position('top')
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        tmpax = ax.twiny()
        tmpax.xaxis.set_ticks([])
        tmpax.xaxis.set_label_position('bottom')
        axs2.append(tmpax)
    axs[0,0].set_ylabel('Weak amplification')
    axs[1,0].set_ylabel('Strong amplification')
    axs[0,0].set_xlabel('Amplifying dark regions')
    axs[0,1].set_xlabel('Amplifying bright regions')
    axs2[0].set_xlabel(t_eqs[0])
    axs2[1].set_xlabel(t_eqs[1])
    axs2[2].set_xlabel(t_eqs[2])
    axs2[3].set_xlabel(t_eqs[3])
    f.savefig('./paper/figures/amplifier_t.png', bbox_inches='tight')

    f2, ax = plt.subplots(num=2)
    f2.suptitle('Source Image')
    #  ax.imshow(resizeforplot(sharpen(I, bg)), interpolation='none')
    ax.imshow(resizeforplot(I), interpolation='none')
    ax.axis('off')
    f2.savefig('./paper/figures/amplifier_I.png', bbox_inches='tight')

    f3, axs = plt.subplots(2,2, num=3)
    f3.suptitle(r'Whole Image Brightening:  $\mathbf{J} = \frac{\mathbf{I}-\mathbf{0}}{\mathbf{t}} + \mathbf{0}$')
    f3b, axs3b = plt.subplots(2,2, num=33)
    f3b.suptitle(r'Whole Image Brightening followed by Sharpening')
    for n,(ax, ax3b, t) in enumerate(zip(axs.ravel(), axs3b.ravel(), [a,b,c,d])):
        J = solveJ(I, 0, t)
        J3b = sharpen(J, bg)
        ax.imshow(resizeforplot(J), interpolation='none')
        ax.set_title(t_eqs[n])
        ax.axis('off')
        ax3b.axis('off')
        ax3b.imshow(resizeforplot(J3b))
    f3.savefig('paper/figures/amplifier_b.png', bbox_inches='tight')
    f3b.savefig('paper/figures/amplifier_b_sharpen.png', bbox_inches='tight')

    f4, axs = plt.subplots(2,2, num=4)
    f4.suptitle(r'Whole Image Darkening:  $\mathbf{J} = \frac{\mathbf{I}-\mathbf{1}}{\mathbf{t}} + \mathbf{1}$')
    f4b, axs4b = plt.subplots(2,2, num=44)
    f4b.suptitle(r'Whole Image Darkening followed by Sharpening')
    for n,(ax, ax4b, t) in enumerate(zip(axs.ravel(), axs4b.ravel(), [a,b,c,d])):
        J = solveJ(I, 1, t)
        J4b = sharpen(J, bg)
        ax.imshow(resizeforplot(J), interpolation='none')
        ax.set_title('%s' %  t_eqs[n])
        ax.axis('off')
        ax4b.axis('off')
        ax4b.imshow(resizeforplot(J4b))
    f4.savefig('paper/figures/amplifier_d.png', bbox_inches='tight')
    f4b.savefig('paper/figures/amplifier_d_sharpen.png', bbox_inches='tight')
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
    #  f5.savefig('paper/figures/compose_parallel_avg.png', bbox_inches='tight')

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
    #  f6.savefig('paper/figures/compose_series_bd.png', bbox_inches='tight')
    #  f7.savefig('paper/figures/compose_series_db.png', bbox_inches='tight')


    plt.show(block=False)
