import cv2
import scipy.ndimage as ndi
import numpy as np
from matplotlib import pyplot as plt
import ietk.util as U
import ietk.methods as M
from ietk.data import IDRiD


# load an image
dset = IDRiD()
#  img, labels = dset['IDRiD_21']
img_id, img, labels = dset.sample()
#  he = labels['HE']
#  ma = labels['MA']
#  ex = labels['EX']
#  se = labels['SE']
#  od = labels['OD']
# set background pure black.
img, fg = U.center_crop_and_get_foreground_mask(img)
bg = ~fg
img[bg] = 0

#  I = img[1500:3000, 1500:3000]
I = img

print('CLAHE')
bgr = (I[:,:,[2,1,0]]*255).astype('uint8')
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=4.0,tileGridSize=(50,50))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
I2 = bgr[:,:,[2,1,0]]

bgr = (I[:,:,[2,1,0]]*255).astype('uint8')
lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
lab_planes = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(5,5))
lab_planes[0] = clahe.apply(lab_planes[0])
lab = cv2.merge(lab_planes)
bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
I3 = bgr[:,:,[2,1,0]]
bgr = cv2.ximgproc.guidedFilter(I.astype('float32'), I3.astype('float32'), 2, 1e-8)

from ietk.methods import sharpen_img
print('A+B+X', img_id)
#  I3 = M.all_methods['A+X'](I, fg)
I3 = sharpen_img.sharpen(I, ~fg, use_guidedfilter=False)

print('plot')
fig, (a,b,c) = plt.subplots(1,3, figsize=(15,15))
a.imshow(I)
a.set_title('Unmodified Image')
b.imshow(I2)
b.set_title('CLAHE, clip limit 3, 5x5 grid')
c.imshow(I3)
c.set_title('Sharpen, Algo. 2 (no GF)')
[x.axis('off') for x in [a,b,c]]
import os
os.makedirs('./data/plots/competing', exist_ok=True)
fig.savefig(f'./data/plots/competing/clahe_vs_sharpen_{img_id}.png', bbox_inches='tight')
