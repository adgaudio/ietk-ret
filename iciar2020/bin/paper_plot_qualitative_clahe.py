from ietk.methods.competing_methods import clahe
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

import paper_plot_qualitative as ppq

img_idx = 416

ns = ppq.params
dct = ns.dset_qualdr[img_idx]
save_fp = f'{ns.save_fig_dir}/qualitative-{img_idx}-clahe-clip.png'
os.makedirs(ns.save_fig_dir, exist_ok=True)

s = time.time()
I = np.array(dct['image']).astype('float')/255.
print(I.shape)
f, axs = plt.subplots(1, 5, num=1, figsize=(4*5, 4))
for ax_idx, clip in enumerate([0.1, 1, 5, 10, 30]):
    print('clahe', clip)
    enhanced_img = clahe(I, clipLimit=clip, tileGridSize=(8,8), colorspace=cv2.COLOR_RGB2LAB)
    axs[ax_idx].imshow(enhanced_img)
    axs[ax_idx].set_title(f"Clip Limit: {clip}", fontsize=20)
    axs[ax_idx].axis('off')
f.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0.1)
f.savefig(save_fp, bbox_inches='tight')
print(time.time()-s)

save_fp = f'{ns.save_fig_dir}/qualitative-{img_idx}-clahe-grid.png'
f, axs = plt.subplots(1, 5, num=2, figsize=(4*5, 4))
for ax_idx, gridsize in enumerate([(8,8), (20,20), (80,80), (150,150), (250,250)]):
    clip = 30
    enhanced_img = clahe(I, clipLimit=clip, tileGridSize=gridsize, colorspace=cv2.COLOR_RGB2LAB)
    axs[ax_idx].imshow(enhanced_img)
    axs[ax_idx].set_title(f"Grid Size: {clip}", fontsize=20)
    axs[ax_idx].axis('off')
f.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0.1)
f.savefig(save_fp, bbox_inches='tight')

s = time.time()
from ietk.methods.sharpen_img import sharpen
from ietk.util import get_background
from skimage.color import rgb2lab, lab2rgb

save_fp = f'{ns.save_fig_dir}/qualitative-{img_idx}-sharpen-nogf.png'
f, axs = plt.subplots(1, 5, num=3, figsize=(4*5, 4))
bg = get_background(I)
I[bg] = 0
#  for ax_idx, t in enumerate([.5, .3, .15, .1, .05]):
for ax_idx, t in enumerate([.1, .01, .005, .002, .001,]):
    print('sharpen l', t)
    labI = rgb2lab(I)
    assert labI.shape[-1] == 3
    labI[:,:,0] = sharpen(labI[:,:,0], t=t, blur_guided_eps=.1, use_guidedfilter=False)
    enhanced_img = lab2rgb(labI)
    axs[ax_idx].imshow(enhanced_img)
    axs[ax_idx].set_title(f"t: {t}", fontsize=20)
    axs[ax_idx].axis('off')
f.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0.1)
f.savefig(save_fp, bbox_inches='tight')
print(time.time() - s)
