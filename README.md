# IETK-Ret - Image Enhancement Toolkit for Retinal Fundus Images

This repository contains a collection of enhancement methods useful for
retinal fundus images, with emphasis on Pixel Color Amplification.

`ietk.util` - methods to separate the fundus from the black background,
as well as crop the image to minimize background.

`ietk.methods` - a set of enhancement methods, mostly based on pixel
color amplification.

`ietk.data` - access to the images in the IDRiD dataset for R&D.


<!-- It also contains the code used for the Pixel Color Amplification paper: -->
<!-- todo -->
<!-- [code](./iciar2020)  [paper: Pixel Color Amplification](TODO) -->


# Usage

```
git clone <this repo>
python setup.py develop

# some example enhanced images
python ietk/methods/sharpen_img.py
python ietk/methods/brighten_darken.py
```


Example usage:
```
from matplotlib import pyplot as plt
from ietk import methods
from ietk import util
from ietk.data import IDRiD

# load an image
dset = IDRiD('./data/IDRiD_segmentation')
img_id, img, labels = dset.sample()
print("using image", img_id)

# crop it and get a focus region
I = img.copy()
I, fg = util.center_crop_and_get_foreground_mask(I)

# enhance the image with some enhancement method
enhanced_img = methods.all_methods['A+B+X'](I, focus_region=fg)

# --> print a list of other enhancements
print(methods.all_methods.keys())

# plot results
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img)
ax2.imshow(enhanced_img)
f.tight_layout()
```


# Disclaimer

This code, including the API and methods, is subject to change without
notice.  If you use it, remember the specific commit you used.
