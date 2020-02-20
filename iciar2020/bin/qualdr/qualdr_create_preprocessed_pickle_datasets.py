"""
Make a copy of arsn dataset that has been enhanced with each method.
Make one copy of dataset per method
"""

import argparse as ap
import ietk.methods
import simplepytorch.api as api
import torchvision.transforms as tvt
from functools import partial
import os.path
import multiprocessing as mp
import pickle

from model_configs.shared_preprocessing import preprocess, pil_to_numpy


def bap():
    par = ap.ArgumentParser()
    A = par.add_argument
    A('--save_dir', default='./data/arsn_preprocess/')
    A('--methods', nargs='+',
      choices=list(ietk.methods.all_methods.keys()),
      default=['identity'])
      # default=list(ietk.methods.all_methods.keys()))
    A('--ok-if-exists', action='store_true')
    return par


def imtrans(method_name):
    return tvt.Compose([
        pil_to_numpy,
        partial(preprocess, method_name=method_name,
                rot=0, flip_y=False, flip_x=False),
    ])


def methods_by_name_and_set(ns):
    for method_name in ns.methods:
        for default_set in ['train', 'test']:
            yield (method_name, default_set)


def gen_dataset(method_name, default_set):
        dset = api.datasets.QualDR(
            default_set=default_set, img_transform=imtrans(method_name),
            getitem_transform=None)
        print('METHOD', method_name)
        new_base_dir = f'./data/preprocessed/arsn_qualdr-ietk-{method_name}/{default_set}/'
        if not ns.ok_if_exists and os.path.exists(new_base_dir):
            raise Exception('dir exists')
        for n, srcfp_idx in enumerate(dset._train_or_test_index):
            print(new_base_dir)
            fp = os.path.join(new_base_dir, os.path.basename(dset._image_loader.fps[srcfp_idx]))
            # skip over paths that exist
            if os.path.exists(fp+'.pickle'):
                print('skip', fp)
                continue
            #  raise Exception('')
            dct = dset[n]

            dct['image'] = dct['image'][0].permute(1,2,0).numpy()*255
        #  for dct in dset:
        #      fp = re.sub('.*?arsn_qualdr/imgs[12]/', new_base_dir, dct['fp'])
            os.makedirs(os.path.dirname(fp), exist_ok=True)
            print(fp)
            assert method_name in fp
            dct['default_set'] = default_set

            #  plt.imshow(dct['image'])
            #  plt.show(block=True)
            #  import sys ; sys.exit()
            print(dct['image'].shape)
            print(dct.keys())
            with open(fp+'.pickle', 'wb') as fout:
                pickle.dump(dct, fout)


if __name__ == "__main__":
    NS = bap().parse_args()

    ns = NS
    with mp.Pool(12) as pool:
        pool.starmap(gen_dataset, methods_by_name_and_set(ns))
