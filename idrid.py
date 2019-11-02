"""
class to load IDRiD dataset
"""
import glob
from os.path import basename
import re
import numpy as np
from matplotlib import pyplot as plt


class IDRiD:
    """Class to load images from the IDRiD Segmentation dataset.
    Details about the data online: https://idrid.grand-challenge.org/

        >>> dset = IDRiD('./data/IDRiD_segmentation')
        >>> img, label_masks = dset['IDRiD_03']
        >>> img.shape, label_masks['HE'].shape, label_masks.keys()
            ((2848, 4288, 3),
            (2848, 4288),
            dict_keys(['imgs', 'MA', 'HE', 'EX', 'SE', 'OD'])))

    Note that not all images display all lesion types.  This means some masks
    will be entirely black.

        >>> print('num images containing each lesion type:', '\n',
                  {k: len(v) for k, v in dset.fps.items()})
            num images containing each lesion type:
             {'imgs': 54, 'MA': 54, 'HE': 53, 'EX': 54, 'SE': 26, 'OD': 54}

    """
    def __init__(self, base_dir='./data/IDRiD_segmentation', train=True):
        """base_dir - (str) filepath to the IDRiD Segmentation dataset"""
        if train is not True:
            raise NotImplementedError('we use training images only for now')

        # get filepaths to images
        _img_fp_globexpr = {
            'imgs': '{base_dir}/1. Original Images/a. Training Set/*',
            #  'labels': '{base_dir}/2. All Segmentation Groundtruths/a. Training Set/**'
            'MA': '{base_dir}/2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms/*',
            'HE': '{base_dir}/2. All Segmentation Groundtruths/a. Training Set/2. Haemorrhages/*',
            'EX': '{base_dir}/2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates/*',
            'SE': '{base_dir}/2. All Segmentation Groundtruths/a. Training Set/4. Soft Exudates/*',
            'OD': '{base_dir}/2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc/*',
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
                if labels=None, assume labels=('HE', 'ME', 'EX', 'SE', 'OD')
        Return:
            img - RGB 3 channel image, normalized between 0 and 1
            labels - dict of form {label: label_mask} where label_mask is a
                boolean array with same height and width as image.
        """
        img = plt.imread(self.fps['imgs'][img_id]) / 255
        if labels is None:
                labels = ('HE', 'ME', 'EX', 'SE', 'OD')

        _empty_label_img = np.zeros(img.shape, dtype='bool')
        masks = {
            label_name: plt.imread(fp_dct[img_id])[:, :, 0].astype('bool')
            if img_id in fp_dct else _empty_label_img.copy()
            for label_name, fp_dct in self.fps.items()}
        return img, masks

    def __getitem__(self, img_id):
        return self.load_img(img_id)


if __name__ == "__main__":
    # for testing
    dset = IDRiD('./data/IDRiD_segmentation')
    img, label_masks = dset['IDRiD_03']
