# IETK-Ret - Image Enhancement Toolkit for Retinal Fundus Images

This repository contains a collection of enhancement methods useful for
retinal fundus images, with emphasis on **Pixel Color Amplification**.

Papers:

- [./iciar2020/](./iciar2020)  -  Code, slides and paper for the **Pixel Color Amplification paper** from ICIAR 2020.

<!-- It also contains the code used for the Pixel Color Amplification paper: -->
<!-- todo -->
<!-- [code](./iciar2020)  [paper: Pixel Color Amplification](TODO) -->

Code:

- `ietk.util` - methods to separate the fundus from the black background,
as well as crop the image to minimize background.
- `ietk.methods` - a set of enhancement methods, mostly based on pixel
color amplification for brightening, darkening and sharpening.
- `ietk.data` - access to the images in the IDRiD dataset for R&D
  (assuming you already downloaded the dataset)


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

# load an image from the IDRiD dataset  (assuming you already have it on disk)
dset = IDRiD('./data/IDRiD_segmentation')
img_id, img, labels = dset.sample()
print("using image", img_id)

# crop fundus image and get a focus region  (a very useful function!)
I = img.copy()
I, fg = util.center_crop_and_get_foreground_mask(I)

# enhance the image with an enhancement method from the ICIAR 2020 paper
# (any combination of letters A,B,C,D,W,X,Y,Z and sA,sB,sC,... are supported)
enhanced_img = methods.brighten_darken['A+B+X'](I, focus_region=fg)
enhanced_img2 = methods.sharpen(enhanced_img, bg=~fg)

# plot results
f, (ax1, ax2, ax3) = plt.subplots(1, 3)
ax1.imshow(img)
ax2.imshow(enhanced_img)
ax3.imshow(enhanced_img2)
f.tight_layout()
```


# Disclaimer

This code, including the API and methods, may not be backwards
compatible between releases.  If you use it, fix the version, git tag or
git commit used in your requirements.txt file.
