import scipy.ndimage as ndi
import numpy as np
from cv2.ximgproc import guidedFilter
from matplotlib import pyplot as plt

from ietk.methods.sharpen_img import sharpen
from ietk import util
from ietk.data import IDRiD


def solvet(I, A):
    z = 1-ndi.minimum_filter((I/A).min(-1), (5, 5))
    return gf(I, z).reshape(*I.shape[:2], 1)


def solvetmax(I, A):
    z = 1-ndi.maximum_filter((I/A).max(-1), (5, 5))
    return gf(I, z).reshape(*I.shape[:2], 1)


def solveJ(I, A, t):
    return (I-A)/t + A


def gf(guide, src, r=100, eps=1e-8):
    return guidedFilter(guide.astype('float32'), src.astype('float32'), r, eps).astype('float64')


if __name__ == "__main__":
    from ietk import methods
    from ietk import metric
    # load an image
    dset = IDRiD('./data/IDRiD_segmentation')
    #  img_id, img, labels = dset.sample()
    img, labels = dset['IDRiD_03']
    #  he = labels['HE']
    #  ma = labels['MA']
    #  ex = labels['EX']
    #  se = labels['SE']
    #  od = labels['OD']
    # set background pure black.

    I = img.copy()
    I = img[1500:2500, 1500:2500, :]

    #  I[:,:,2] = I[:,:,(0,1)].mean(-1) # ignore the third channel.
    I[:,:,2] = I[:,:,1]  # ignore the third channel.

    bg = np.zeros_like(I, dtype='bool')
    #  bg = util.get_background(I)
    #  I[bg] = 0
    #  Imask = labels['MA'][1500:3000, 1500:3000]

    # four transmission maps
    a = solvet(1-I, 1)  # == 1-solvetmax(I, 1)
    d = 1-solvet(1-I, 1)  # == solvetmax(I, 1)
    c = 1-solvet(I, 1)  # == solvetmax(1-I, 1)
    b = solvet(I, 1)  # == 1-solvetmax(1-I, 1)

    # Brighten
    A, B, C, D = [solveJ(I, 0, t) for t in [a,b,c,d]]
    # Darken
    W, X, Y, Z = [solveJ(I, 1, t) for t in [a,b,c,d]]

    # Plots
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
        ax.imshow(t.squeeze(), cmap='gray', interpolation='none')
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
    ax.imshow(sharpen(I, bg), interpolation='none')
    ax.imshow(I, interpolation='none')
    ax.axis('off')
    f2.savefig('./paper/figures/amplifier_I.png', bbox_inches='tight')

    f3, axs = plt.subplots(2,2, num=3)
    f3.suptitle(r'Whole Image Brightening:  $\mathbf{J} = \frac{\mathbf{I}-\mathbf{0}}{\mathbf{t}} + \mathbf{0}$')
    for n,(ax, t) in enumerate(zip(axs.ravel(), [a,b,c,d])):
        J = solveJ(I, 0, t)
        #  J = sharpen(J, bg)
        ax.imshow(J, interpolation='none')
        ax.set_title(t_eqs[n])
        ax.axis('off')
    f3.savefig('paper/figures/amplifier_b.png', bbox_inches='tight')

    f4, axs = plt.subplots(2,2, num=4)
    f4.suptitle('Darken')
    for n,(ax, t) in enumerate(zip(axs.ravel(), [a,b,c,d])):
        J = solveJ(I, 1, t)
        #  J = sharpen(J, bg)
        ax.imshow(J, interpolation='none')
        ax.set_title('%s' %  t_eqs[n])
        ax.axis('off')
    f4.suptitle(r'Whole Image Darkening:  $\mathbf{J} = \frac{\mathbf{I}-\mathbf{1}}{\mathbf{t}} + \mathbf{1}$')
    f4.savefig('paper/figures/amplifier_d.png', bbox_inches='tight')

    plt.show(block=False)
