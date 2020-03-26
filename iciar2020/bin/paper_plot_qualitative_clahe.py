from ietk.methods.competing_methods import clahe
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

import paper_plot_qualitative as ppq

img_idx = 416

ns = ppq.params
dct = ns.dset_qualdr[img_idx]
save_fp = f'{ns.save_fig_dir}/qualitative-{img_idx}-clahe-clip.png'
os.makedirs(ns.save_fig_dir, exist_ok=True)

I = np.array(dct['image']).astype('float')/255.
print(I.shape)
f, axs = plt.subplots(1, 5, num=1, figsize=(4*5, 4))
for ax_idx, clip in enumerate([0.1, 0.5, 1, 4, 10]):
    enhanced_img = clahe(I, clipLimit=clip, tileGridSize=(8,8), colorspace=cv2.COLOR_RGB2LAB)
    axs[ax_idx].imshow(enhanced_img)
    axs[ax_idx].set_title(f"Clip Limit: {clip}", fontsize=20)
    axs[ax_idx].axis('off')
f.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0.1)
f.savefig(save_fp, bbox_inches='tight')

save_fp = f'{ns.save_fig_dir}/qualitative-{img_idx}-clahe-grid.png'
f, axs = plt.subplots(1, 5, num=2, figsize=(4*5, 4))
for ax_idx, gridsize in enumerate([(2,2), (5,5), (8,8), (20,20), (30,30)]):
    clip = 1
    enhanced_img = clahe(I, clipLimit=clip, tileGridSize=gridsize, colorspace=cv2.COLOR_RGB2LAB)
    axs[ax_idx].imshow(enhanced_img)
    axs[ax_idx].set_title(f"Grid Size: {clip}", fontsize=20)
    axs[ax_idx].axis('off')
f.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0.1)
f.savefig(save_fp, bbox_inches='tight')
