#!/usr/bin/env python
"""
On IDRiD dataset,
Generate histograms of pixel intensities for the red, green and blue channels
after pre-processing methods are applied.
Save histogram data and plots to file.
"""
import numpy as np
from matplotlib import pyplot as plt
from os.path import join
import os

from ietk import methods
#  from ietk.data import IDRiD
from ietk import util

import model_configs.shared_preprocessing as SP
import simplepytorch.datasets as D


def sep_pixels(img, focus_region, mask):
    assert focus_region.shape == (img.shape[0], img.shape[1])
    diseased = img[focus_region & mask]
    healthy = img[focus_region & ~mask]
    return healthy, diseased


def iter_imgs(ns):
    #  dset = IDRiD(ns.data_dir)


    dset = D.IDRiD_Segmentation(getitem_transform=lambda dct: (
        dct['img_id'], D.IDRiD_Segmentation.as_tensor(return_numpy_array=True)(dct)))
    for img_id, tensor in dset:
        img, _stack = SP.preprocess(
            tensor, 'identity', rot=0, flip_x=0, flip_y=0)
        _stack = _stack.numpy().astype('bool')
        img = img.permute(1,2,0).numpy()
        labels = {k: _stack[i] for i, k in enumerate(D.IDRiD_Segmentation.LABELS)}

        focus_region = util.get_foreground(img)
        for method_name, func in methods.all_methods.items():
            if method_name not in ns.methods:
                continue
            modified_img = None
            for lesion_name, mask in labels.items():
                modified_img = func(img=img, focus_region=focus_region)
                try:
                    healthy, diseased = sep_pixels(
                        modified_img, focus_region, mask)
                except Exception:
                    print("FAIL", img_id, lesion_name, method_name, type(img), type(focus_region))
                    raise
                yield (img_id, method_name, lesion_name, healthy, diseased)


def gen_histogram(img_id, method_name, lesion_name, healthy, diseased, **junk):
    #  for ch in [0,1,2]:
        #  plt.hist(healthy[:, ch], bins=256, range=(0,1), density=True, label='healthy', alpha=.4)
        #  plt.hist(diseased[:, ch], bins=256, range=(0,1), density=True, label='diseased', alpha=.4)
        #  plt.title(f'{img_id} {lesion_name} {method_name} -- channel {ch}')
        #  plt.legend()
        #  plt.show()
    color = {0: 'red', 1: 'green', 2: 'blue'}
    fig, axs = plt.subplots(3, 1, num=1)

    for ax, ch in zip(axs.ravel(), [0,1,2]):
        ax.hist(healthy[:, ch].clip(0-1/255,1+1/255)*255, bins=256+2, range=(-1,257), density=True, label='healthy', alpha=.3, color=color[ch], lw=0)
        ax.hist(diseased[:, ch].clip(0-1/255,1+1/255)*255, bins=256+2, range=(-1,257), density=True, label='diseased', alpha=.7, color=color[ch], lw=0)
        ax.legend()
        ax.set_title(f'channel: {color[ch]}')
    fig.suptitle(f'{img_id} {lesion_name} {method_name}')
    return fig


def main(ns):
    os.makedirs(ns.save_dir, exist_ok=True)
    os.makedirs(ns.save_dir_data, exist_ok=True)
    for (img_id, method_name, lesion_name, healthy, diseased) in iter_imgs(ns):
        filename_prefix = f'{img_id}-{lesion_name.replace(" ","_")}-{method_name.replace(" ","_")}'
        img_fp = join(ns.save_dir, f'{filename_prefix}.png')
        data_fp = join(ns.save_dir_data, f'{filename_prefix}.npz')
        data_3d_fp = join(ns.save_dir_data, f'hist3d-{filename_prefix}.npz')
        #  if os.path.exists(img_fp):
            #  print(f'Image file exists, not re-creating: {img_fp}')
        #  else:
            #  fig = gen_histogram(**locals())
            #  fig.savefig(img_fp)
            #  if ns.show_plot:
                #  plt.show()
            #  else:
                #  plt.close(fig)
        if os.path.exists(data_fp):
            print(f'Data file exists, not re-creating: {data_fp}')
        else:
            hs, ds = [], []
            for ch in [0,1,2]:
                h, _ = np.histogram(healthy[:, ch].clip(0-1/255,1+1/255)*255, bins=256+2, range=(-1,257))
                d, _ = np.histogram(diseased[:, ch].clip(0-1/255,1+1/255)*255, bins=256+2, range=(-1,257))
                hs.append(h)
                ds.append(d)
            np.savez_compressed(data_fp, healthy=np.stack(hs), diseased=np.stack(ds))
        #  if os.path.exists(data_3d_fp):
            #  print(f'3d histogram data file exists, not re-creating: {data_3d_fp}')
        #  else:
            #  H = np.histogramdd(healthy.clip(0-1/255,1+1/255), bins=256+2, range=[(-1,257)]*3)[0]
            #  D = np.histogramdd(diseased.clip(0-1/255,1+1/255), bins=256+2, range=[(-1,257)]*3)[0]
            #  np.savez_compressed(data_3d_fp, healthy=H, diseased=D)


def bap():
    p = ap.ArgumentParser()
    p.add_argument('--img-ids', nargs='*', help='By default, analyze all imgs.  If this parameter is given, only analyze given imgs:  --img-ids IDRiD_01 IMDiD_02')
    p.add_argument('--labels', nargs='*', default=('MA', 'HE', 'EX', 'SE', 'OD'), help='Lesions to analyze', choices=('MA', 'HE', 'EX', 'SE', 'OD'))
    p.add_argument('--methods', nargs='*', default=tuple(methods.all_methods.keys()), choices=tuple(methods.all_methods.keys()), help='list of methods named in methods.all_methods')
    p.add_argument('--data-dir', default='./data/IDRiD_segmentation', help='Location of the IDRiD dataset')
    p.add_argument('--save-dir', default='./data/histograms_idrid_plots', help='Directory to save generated histograms')
    p.add_argument('--save-dir-data', default='./data/histograms_idrid_data', help='Directory to save generated plots')
    p.add_argument('--show-plot', action='store_true', help='by default, dont show plot.  just save a figure to disk')
    return p


if __name__ == "__main__":
    # use like this
    # python  gen_histograms.py --labels MA HE --methods "Unmodified Image" "Sharpen, t=0.15" --img-ids IDRiD_01 IDRiD_02
    # or faster:
    # seq -f '%02g' 54 -1 1 | parallel -j 8 python bin/gen_histograms.py --img-ids IDRiD_{}
    import argparse as ap
    NS = bap().parse_args()
    print(NS)
    main(NS)
