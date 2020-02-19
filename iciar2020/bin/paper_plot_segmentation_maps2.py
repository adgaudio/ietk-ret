from matplotlib import pyplot as plt
import numpy as np
import PIL
import cv2
import model_configs.shared_preprocessing as SP
import ietk.util as U


def open(fp, ch):
    im = cv2.imread(fp)
    if ch == 1:
        return im[:,:,-1]
    else:
        return im[:,:,[2,1,0]]


orig = open('./data/RITE/test/images/01_test.tif', 3)
gt_segm = np.array(PIL.Image.open('./data/RITE/test/vessel/01_test.png'))

orig, _, gt_segm = U.center_crop_and_get_foreground_mask(
    orig, is_01_normalized=False, label_img=gt_segm)
gt_segm = SP.affine_transform(np.dstack([gt_segm, gt_segm, gt_segm]),
                              rot=0, flip_x=0, flip_y=0)[:,:,0]



enhanced = open('data/results/Rtest2-sC+sX/images/88-0-input.tiff', 3)
segm = np.dstack([
    open('data/results/Rtest2-sC+sX/images/88-0-vessels.tiff', 1),
    open('data/results/Rtest2-sC+sX/images/88-0-overlap.tiff', 1),
    open('data/results/Rtest2-sC+sX/images/88-0-arteries.tiff', 1),
    open('data/results/Rtest2-sC+sX/images/88-0-veins.tiff', 1),
]).astype('bool')
enh_segm = segm.sum(-1).astype('bool')

identity = open('data/results/Rtest2-identity/images/97-0-input.tiff', 3)
segm = np.dstack([
    open('data/results/Rtest2-identity/images/97-0-vessels.tiff', 1),
    open('data/results/Rtest2-identity/images/97-0-overlap.tiff', 1),
    open('data/results/Rtest2-identity/images/97-0-arteries.tiff', 1),
    open('data/results/Rtest2-identity/images/97-0-veins.tiff', 1),
]).astype('bool')
id_segm = segm.sum(-1).astype('bool')


fig, axs = plt.subplots(1, 4, figsize=(20,20))
axs[0].set_title('sC+sX vs Ground Truth', fontsize=22)
axs[0].imshow(np.dstack([enh_segm*255, gt_segm, gt_segm]), interpolation='none')
axs[1].set_title('sC+sX Enhancement', fontsize=22)
axs[1].imshow(enhanced, interpolation='none')
axs[2].set_title('Unmodified Image', fontsize=22)
axs[2].imshow(orig, interpolation='none')
axs[3].set_title('sC+sX vs Identity', fontsize=22)
axs[3].imshow(np.dstack([enh_segm*255, id_segm*255, id_segm*255]), interpolation='none')
[x.axis('off') for x in axs]
fig.savefig('./data/plots/rite_segmentation.png', bbox_inches='tight')
#  plt.show()
#  plt.pause(2)
