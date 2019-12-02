import numpy as np
from matplotlib import pyplot as plt


dat = np.load('data/histograms_idrid_data/IDRiD_39-MA-Illuminate_Sharpen.npz')
h = dat['healthy']
d = dat['diseased']
hs = (h/h.sum()).cumsum()
ds = (d/d.sum()).cumsum()

plt.plot(np.arange(256), hs[:256], 'r', label='healthy, red channel', alpha=.7)
plt.plot(np.arange(256,256*2), hs[256:256*2], 'g', label='healthy, green channel', alpha=.7)
plt.plot(np.arange(256*2,256*3), hs[256*2:], 'b', label='healthy, blue channel', alpha=.7)

plt.plot(np.arange(256), ds[:256], 'r', label='diseased, red channel', alpha=.3)
plt.plot(np.arange(256,256*2), ds[256:256*2], 'g', label='diseased, green channel', alpha=.3)
plt.plot(np.arange(256*2,256*3), ds[256*2:], 'b', label='diseased, blue channel', alpha=.3)

idx = np.abs(hs - ds).argmax()
plt.vlines(idx, min(hs[idx], ds[idx]), max(hs[idx], ds[idx]), label='KS score (max abs difference)')
plt.legend()
plt.savefig('./ks_example.png')
plt.show()
