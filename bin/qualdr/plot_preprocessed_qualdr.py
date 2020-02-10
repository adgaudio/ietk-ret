
# view images from a pickled dicts dataset
# compare to identity
import os
import os.path as P
import re
import sys
import random
import pickle
from matplotlib import pyplot as plt


dirp = sys.argv[1]
fns = os.listdir(dirp)
random.shuffle(fns)

f, (ax1, ax2) = plt.subplots(1, 2)

for fname in fns:
    fp = P.join(dirp, fname)
    with open(fp, 'rb') as fin:
        dct = pickle.load(fin)
    print(dct['image'].max(), dct['image'].min())
    ax1.imshow(dct['image'])
    ifp = re.sub(r'(arsn_qualdr-ietk-)(.*?)(/)', r'\1identity\3', fp)
    print(ifp)
    with open(ifp, 'rb') as ifin:
        idct = pickle.load(ifin)
    identity_img = idct['image']
    ax2.imshow(identity_img.astype('uint8'))
    #  plt.pause(0.5)
    plt.waitforbuttonpress()
