import os
import os.path as P
import re
import sys
import random
import pickle
from matplotlib import pyplot as plt
dirp = sys.argv[1]
fns = os.listdir(dirp)
for fdir in fns:
    f, ax = plt.subplots(1,1)
    if 'arsn' not in fdir: continue
    fp = P.join(dirp, fdir, 'train/1.2.392.200046.100.3.8.103441.9367.20170207114824.2.1.1.1.png.pickle')
    with open(fp, 'rb') as fin:
        dct = pickle.load(fin)
    print(dct['image'].max(), dct['image'].min())
    ax.set_title(fdir)
    ax.imshow(dct['image'])
    print(fp)
    #  plt.pause(0.5)
    plt.waitforbuttonpress()
