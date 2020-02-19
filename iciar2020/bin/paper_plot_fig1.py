import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_grid import ImageGrid
import competing_methods
import util

    #  STYLE FOR A POSTER presentation
SMALL_SIZE = 30
MEDIUM_SIZE = 45
BIGGER_SIZE = 55
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE, figsize=(20,20))  # fontsize of the figure title
plt.rc('lines', linewidth=10, markersize=25)

#  dat = np.load('data/histograms_idrid_data/IDRiD_39-MA-Illuminate_Sharpen.npz')
#  h = dat['healthy']
#  d = dat['diseased']
#  hs = (h/h.sum()).cumsum()
#  ds = (d/d.sum()).cumsum()


img_id = 'IDRiD_03'
im = plt.imread(f'data/IDRiD_segmentation/1. Original Images/a. Training Set/{img_id}.jpg')
im2 = competing_methods.illuminate_sharpen(im, focus_region=util.get_foreground(im))
fig, axs = plt.subplots(1, 2, figsize=(8,4))
fig = plt.figure(figsize=(20,20))
axs = ImageGrid(fig, 111, (1, 2), share_all=True)
axs[0].get_yaxis().set_ticks([])
axs[0].get_xaxis().set_ticks([])
axs[0].imshow(im)
axs[1].imshow(im2)  # TODO
fig.savefig(f'./paper_plot_fig1_{img_id}.png', bbox_inches='tight')
fig, ax = plt.subplots(figsize=(20,20))
ax.imshow(im2)
ax.axis('off')
fig.savefig(f'{img_id}_illuminate_sharpen.png',  bbox_inches="tight")
ax.imshow(im)
ax.axis('off')
fig.savefig(f'{img_id}.png', bbox_inches='tight')
