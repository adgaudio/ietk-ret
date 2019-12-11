"""
plots of histograms for poster.
"""
import numpy as np
from matplotlib import pyplot as plt

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

dat = np.load('data/histograms_idrid_data/IDRiD_39-MA-Illuminate_Sharpen.npz')
h = dat['healthy']
d = dat['diseased']
hs = (h/h.sum()).cumsum()
ds = (d/d.sum()).cumsum()

fig, ax = plt.subplots(figsize=(20,20))
ax.plot(np.arange(256), hs[:256], 'r', label='healthy, red channel', alpha=.7)
ax.plot(np.arange(256,256*2), hs[256:256*2], 'g', label='healthy, green channel', alpha=.7)
ax.plot(np.arange(256*2,256*3), hs[256*2:], 'b', label='healthy, blue channel', alpha=.7)

ax.plot(np.arange(256), ds[:256], 'r', label='diseased, red channel', alpha=.3)
ax.plot(np.arange(256,256*2), ds[256:256*2], 'g', label='diseased, green channel', alpha=.3)
ax.plot(np.arange(256*2,256*3), ds[256*2:], 'b', label='diseased, blue channel', alpha=.3)

idx = np.abs(hs - ds).argmax()
ax.vlines(idx, min(hs[idx], ds[idx]), max(hs[idx], ds[idx]), label='KS score (max abs difference)')
ax.legend(loc='lower right')
fig.savefig('./ks_example.png')
#  plt.show()
