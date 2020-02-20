"""
Make the images (and masks) smaller.  Useful because the resizing operation
is computationally heavy.
"""
import pickle
import PIL
import numpy as np
import os.path
import torch

from ietk.data import IDRiD
import simplepytorch.datasets as D
import ietk.util as U
import model_configs.segmentation as S

def resize_mask(im):
    return np.array(PIL.Image.fromarray(im).resize((512,512))).astype('bool')


for train_or_test in ['train', 'test']:
    dset = D.IDRiD_Segmentation(
        use_train_set=train_or_test=='train',
        getitem_transform=lambda dct: (dct['img_id'], D.IDRiD_Segmentation.as_tensor(return_numpy_array=True)(dct)))
    for n, (img_id, tensor) in enumerate(dset):
        im, mask = S.preprocess(tensor, 'identity', rot=0, flip_x=False, flip_y=False)
        dct = {'tensor': torch.cat([im, mask]).permute(1,2,0).numpy() * 255}

        # --> debug plots
        #  from matplotlib import pyplot as plt
        #  plt.ion()
        #  #  im_np = dct['tensor'][:,:,:3]
        #  im_np_orig = (tensor[:,:,:3])
        #  plt.imshow(dct['tensor'].permute(1,2,0).numpy()[:,:,[0,1,-1]])
        #  # plt.imshow(im_np_orig.astype('uint8'))
        #  plt.pause(0.01)
        #  continue

        save_fp = f'./data/preprocessed/idrid-identity/{train_or_test}/{img_id}.pickle'
        os.makedirs(os.path.dirname(save_fp), exist_ok=True)
        with open(save_fp, 'wb') as fout:
            pickle.dump(dct, fout)


