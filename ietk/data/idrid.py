"""
class to load IDRiD dataset
"""
import random
import glob
import numpy as np
from os.path import basename
import PIL
import re
from matplotlib import pyplot as plt


class IDRiD:
    """Class to load images from the IDRiD Segmentation dataset.
    Details about the data online: https://idrid.grand-challenge.org/

        >>> dset = IDRiD('./data/IDRiD_segmentation')
        >>> img, label_masks = dset['IDRiD_03']
        >>> img.shape, label_masks['HE'].shape, label_masks.keys()
            ((2848, 4288, 3),
            (2848, 4288),
            dict_keys(['MA', 'HE', 'EX', 'SE', 'OD'])))

    Note that not all images contain all lesion types.  This means some masks
    may be missing from those requested.

        >>> print('num images containing each lesion type:', '\n',
                  {k: len(v) for k, v in dset.fps.items()})
            num images containing each lesion type:
             {54, 'MA': 54, 'HE': 53, 'EX': 54, 'SE': 26, 'OD': 54}

    Label names available in this dataset:
    'MA' - Microaneurysms
    'HE' - Hemorrhages
    'EX' - Hard Exudates
    'SE' - Soft Exudates (Cotton Wool Spots)
    'OD' - Optic Disc Segmentation (not used for Diabetic Retinopathy)
    """
    labels = ('MA', 'HE', 'EX', 'SE', 'OD')
    def __init__(self, base_dir='./data/IDRiD_segmentation', train=True,
                 convert_pil_to_numpy=True):
        """base_dir - (str) filepath to the IDRiD Segmentation dataset"""
        if train:
            train_or_test_dirname = 'a. Training Set'
        else:
            train_or_test_dirname = 'b. Testing Set'
        self.convert_pil_to_numpy = convert_pil_to_numpy

        # get filepaths to images
        _img_fp_globexpr = {
            'imgs': f'{base_dir}/1. Original Images/{train_or_test_dirname}/*',
            #  'labels': '{base_dir}/2. All Segmentation Groundtruths/{train_or_test_dirname}/**'
            'MA': f'{base_dir}/2. All Segmentation Groundtruths/{train_or_test_dirname}/1. Microaneurysms/*',
            'HE': f'{base_dir}/2. All Segmentation Groundtruths/{train_or_test_dirname}/2. Haemorrhages/*',
            'EX': f'{base_dir}/2. All Segmentation Groundtruths/{train_or_test_dirname}/3. Hard Exudates/*',
            'SE': f'{base_dir}/2. All Segmentation Groundtruths/{train_or_test_dirname}/4. Soft Exudates/*',
            'OD': f'{base_dir}/2. All Segmentation Groundtruths/{train_or_test_dirname}/5. Optic Disc/*',
        }
        self.fps = {  # format: {'MA: {'IDRiD_41': './path/to/image'}}
            k: {re.sub(r'(IDRiD_\d{2}).*', r'\1', basename(imgfp)): imgfp
                for imgfp in sorted(glob.glob(v.format(base_dir=base_dir)))}
            for k, v in _img_fp_globexpr.items()}

    def load_img(self, img_id, labels=None):
        """
        Load image and label masks.

        Input:
            img_id: str of form 'IDRiD_XX'  where XX are digits
            labels: list of str defining which label masks to get.
                if labels=None, assume labels=('HE', 'MA', 'EX', 'SE', 'OD')
        Return:
            img - RGB 3 channel image, normalized between 0 and 1
            labels - dict of form {label: label_mask} where label_mask is a
                boolean array with same height and width as image.
        """
        with PIL.Image.open(self.fps['imgs'][img_id]) as img:
            img.load()
            if self.convert_pil_to_numpy:
                img = np.array(img) / 255
        if labels is None:
                labels = {'HE', 'MA', 'EX', 'SE', 'OD'}

        # --> if image has no MA, set the mask as all False.
        #  _empty_label_img = np.zeros(img.shape, dtype='bool')
        if labels:
            masks = {
                label_name: plt.imread(fp_dct[img_id])[:, :, 0].astype('bool')
                #if img_id in fp_dct else _empty_label_img.copy()
                for label_name, fp_dct in self.fps.items()
                if label_name in labels
                and img_id in fp_dct  # ie. ignore MA if image has no MA
            }
        else:
            masks = {}
        return img, masks

    def iter_imgs(self, labels=None, shuffle=False, subset=None):
        """
        Return iterator over the set of images.  For instance:

            >>> dset = IDRiD()
            >>> for img_id, img, _ in dset.iter_imgs(labels=()): break
            >>> for img_id, img, labels in dset.iter_imgs(): break

        Input:
            labels: list of str defining which label masks to get.
                if labels=None, assume labels=('HE', 'MA', 'EX', 'SE', 'OD')
            shuffle: bool. Whether to return images in randomized order.
            subset: list of img ids.  If not given, use all imgs
        Return:
            an iterator over images in the dataset.
        """
        if subset:
            img_ids = list(subset)
            for x in img_ids:
                assert x in self.fps['imgs'], f'unrecognized image id: {x}'
        else:
            img_ids = list(self.fps['imgs'])
        if shuffle:
            random.shuffle(img_ids)
        for img_id in img_ids:
            rv = [img_id]
            rv.extend(self.load_img(img_id, labels))
            yield tuple(rv)

    def sample(self, labels=None):
        img_id = random.choice(list(self.fps['imgs']))
        return (img_id, ) + self.load_img(img_id, labels=labels)

    def __iter__(self):
        yield from self.iter_imgs()

    def __getitem__(self, img_id):
        return self.load_img(img_id)


if __name__ == "__main__":
    # for testing
    dset = IDRiD('./data/IDRiD_segmentation')
    img, label_masks = dset['IDRiD_03']
