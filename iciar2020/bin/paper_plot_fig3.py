import scipy.ndimage as ndi
import numpy as np
from cv2.ximgproc import guidedFilter
from matplotlib import pyplot as plt
from ietk.data import IDRiD
from ietk import util
from ietk import metric
from ietk.methods.sharpen_img import sharpen


def solvet(I, A):
    z = 1-ndi.minimum_filter((I/A).min(-1), (5, 5))
    return gf(I, z).reshape(*I.shape[:2], 1)

def solvetmax(I, A):
    z = 1-ndi.maximum_filter((I/A).max(-1), (5, 5))
    return gf(I, z).reshape(*I.shape[:2], 1)

def solvetmedian(I, A):
    z = 1-ndi.median_filter((I/A).mean(-1), (5, 5))
    return gf(I, z).reshape(*I.shape[:2], 1)


def solveJ(I, A, t):
    return (I-A)/t + A


def gf(guide, src, r=100, eps=1e-8):
    return guidedFilter(guide.astype('float32'), src.astype('float32'), r, eps).astype('float64')


def add_ks(I, bg, L, ax):
    return
    strng = ''
    for lesion, mask in L.items():
        ks = metric.ks_test_max_per_channel(I, mask, ~bg)
        strng += f'{lesion}: {ks} \n'
    ax.text(2, 6, strng, fontsize=8)
    print(strng)
    #  pass


dset = IDRiD('./data/IDRiD_segmentation')
img, labels = dset['IDRiD_02']
bg = util.get_background(img)

I = img#[800:1900, 1700:3800, :]
bg = bg#[800:1900, 1700:3800, :]
L = labels#[800:1900, 1700:3800]
#  I = img[800:1900, 1700:3800, :]
#  bg = bg[800:1900, 1700:3800, :]
#  L = {k: v[800:1900, 1700:3800] for k, v in labels.items()}
I[:,:,-1] = I[:,:,1]
# use a larger neighborhood size for these images:
#  I = plt.imread('./data/brightdark.jpg') ; I = I/I.max()
#  I = plt.imread('./data/spots.jpg') ; I = I/I.max()
#  I = plt.imread('./data/watercave.jpeg') ; I = I/I.max()
#  I = plt.imread('./data/lowcontrastflower.jpeg') ; I = I/I.max()
#  I = plt.imread('./data/citynight.jpg') ; I = I/I.max()
#  I = I/2 + (np.mgrid[:I.shape[0], :I.shape[1]][1]/I.shape[1]/2)[:,:,np.newaxis]
a = solvet(1-I, 1)  # == 1-solvetmax(I, 1)
d = 1-solvet(1-I, 1)  # == solvetmax(I, 1)

c = 1-solvet(I, 1)  # == solvetmax(1-I, 1)
b = solvet(I, 1)  # == 1-solvetmax(1-I, 1)

#  a = a-a.mean()+.5
#  b = b-b.mean() +.5
#  c=c-c.mean()+.5
#  d=d-d.mean()+.5

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
kws = dict(bt='\mathbf{t}', bI='\mathbf{I}', bA='\mathbf{A}')
axs2[0].set_xlabel(r'${bt} = $solve_t$(1-{bI}, {bA}=1)$'.format(**kws))
axs2[1].set_xlabel(r'${bt} = $solve_t$({bI}, {bA}=1)$'.format(**kws))
axs2[2].set_xlabel(r'${bt} = 1-$solve_t$({bI}, {bA}=1)$'.format(**kws))
axs2[3].set_xlabel(r'${bt} = 1-$solve_t$(1-{bI}, {bA}=1)$'.format(**kws))
f.savefig('./paper/figures/amplifier_t.png')


f2, ax = plt.subplots(num=2)
f2.suptitle('Source Image')
ax.imshow(sharpen(I, bg), interpolation='none')
ax.axis('off')

f2.savefig('./paper/figures/amplifier_I.png')
add_ks(sharpen(I, bg), bg, L, ax)


f3, axs = plt.subplots(2,2, num=3)
f3.suptitle(r'Whole Image Brightening:  $\mathbf{J} = \frac{\mathbf{I}-\mathbf{0}}{\mathbf{t}} + \mathbf{0}$')
for n,(ax, t) in enumerate(zip(axs.ravel(), [a,b,c,d])):
    J = solveJ(I, 0, t)
    J = sharpen(J, bg)
    ax.imshow(J, interpolation='none')
    ax.set_title(n)
    ax.axis('off')
    add_ks(J, bg, L, ax)
f3.savefig('paper/figures/amplifier_b.png')

f4, axs = plt.subplots(2,2, num=4)
f4.suptitle('Darken')
for n,(ax, t) in enumerate(zip(axs.ravel(), [a,b,c,d])):
    J = solveJ(I, 1, t)
    J = sharpen(J, bg)
    ax.imshow(J, interpolation='none')
    ax.set_title('%s' %  n)
    ax.axis('off')
    add_ks(J, bg, L, ax)
f4.suptitle(r'Whole Image Darkening:  $\mathbf{J} = \frac{\mathbf{I}-\mathbf{1}}{\mathbf{t}} + \mathbf{1}$')
f4.savefig('paper/figures/amplifier_d.png')

# combine corresponding pairs
# brighten
A, B, C, D = [solveJ(I, 0, t) for t in [a,b,c,d]]
# DARKEN
W, X, Y, Z = [solveJ(I, 1, t) for t in [a,b,c,d]]
A2, B2, C2, D2 = [solveJ(i, 0, t) for i, t in zip([X,X,X,X], [a,b,c,d])]
W2, X2, Y2, Z2 = [solveJ(i, 1, t) for i, t in zip([A,A,A,A], [a,b,c,d])]
WC, XC, YC, ZC = [solveJ(i, 1, t) for i, t in zip([C,C,C,C], [a,b,c,d])]
#  AB  WX
#  CD  YZ
for num, (title, imgs) in enumerate([
        ('(A+X)/2', [A/2+X/2, B/2+Y/2, C/2+X/2, D/2+W/2]),
        ('(AX+XA)/2, (A+X)/2, A2, X2', [A2/2+X2/2, (A+X)/2, A2, X2]),
        ('WC, XC, YC, ZC', [WC, XC, YC, ZC]),
    ]):
        #  ('A o X', [A2, B2, C2, D2]),
        #  ('X o A', [W2, X2, Y2, Z2])]):
    f, axs = plt.subplots(2,2, num=5+num)
    f.clf()
    f, axs = plt.subplots(2,2, num=5+num)
    f.suptitle('Compose: %s' % title)
    for n,(ax, z) in enumerate(zip(axs.ravel(), imgs)):
        z = sharpen(z, bg)
        ax.imshow(z, cmap='gray', interpolation='none')
        #  ax.imshow(util.norm01(z, bg), cmap='gray', interpolation='none')
        ax.set_title(n)
        ax.axis('off')
        add_ks(z, bg, L, ax)


# A+X / 2 is the best one for MA.  A and X are the best for MA
# vasculature
#  z = guidedFilter(I.astype('float32'), (C).astype('float32'), 2, 1e-8)
#  y = sharpen(z, bg)
# choroidal vasculature.
#  x = guidedFilter(I.astype('float32'), (Z).astype('float32'), 2, 1e-2)
#  w = sharpen(x, bg, t=0.15)
#  plt.imshow(w)

# hard exudate segmentation
#  plt.figure() ; plt.imshow((sharpen(Z, bg)>0).all(-1)*255) ; plt.show(block=False)

plt.show(block=False)
