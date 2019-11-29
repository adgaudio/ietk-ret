import numpy as np
from matplotlib import pyplot as plt
from functools import lru_cache
from os.path import join
import os


import competing_methods
import metric
from idrid import IDRiD
import util


def sep_pixels(img, focus_region, mask):
    assert np.all(focus_region[:,:,0] == focus_region[:,:,1])
    assert np.all(focus_region[:,:,2] == focus_region[:,:,1])
    diseased = img[focus_region[:,:,0] & mask]
    healthy = img[focus_region[:,:,0] & ~mask]
    return healthy, diseased


def iter_imgs(dset, ns):
    for img_id, img, labels in dset.iter_imgs(labels=ns.labels, subset=ns.img_ids):
        focus_region = util.get_foreground(img)
        for method_name, func in competing_methods.all_methods.items():
            if method_name not in ns.methods:
                continue
            modified_img = None
            for lesion_name, mask in labels.items():
                if method_name.startswith('Bayes Sharpen, '):
                    # hack: this method requires the lesion name to load the appropriate prior.
                    modified_img = func(img=img, focus_region=focus_region, label_name=lesion_name)
                    print('run')
                elif modified_img is None:
                    modified_img = func(img=img, focus_region=focus_region)
                    print('run')
                healthy, diseased = sep_pixels(
                    modified_img, focus_region, mask)
                yield (img_id, method_name, lesion_name, healthy, diseased)


def main(ns):
    dset = IDRiD(ns.data_dir)
    os.makedirs(ns.save_dir, exist_ok=True)
    for (img_id, method_name, lesion_name, healthy, diseased) in iter_imgs(dset, ns):
        #  for ch in [0,1,2]:
            #  plt.hist(healthy[:, ch], bins=256, range=(0,1), density=True, label='healthy', alpha=.4)
            #  plt.hist(diseased[:, ch], bins=256, range=(0,1), density=True, label='diseased', alpha=.4)
            #  plt.title(f'{img_id} {lesion_name} {method_name} -- channel {ch}')
            #  plt.legend()
            #  plt.show()
        color = {0: 'red', 1: 'green', 2: 'blue'}
        fig, axs = plt.subplots(3, 1)

        for ax, ch in zip(axs.ravel(), [0,1,2]):
            ax.hist(healthy[:, ch], bins=256, range=(0,1), density=True, label='healthy', alpha=.3, color=color[ch], lw=0)
            ax.hist(diseased[:, ch], bins=256, range=(0,1), density=True, label='diseased', alpha=.7, color=color[ch], lw=0)
            ax.legend()
            ax.set_title(f'channel: {color[ch]}')
        fig.suptitle(f'{img_id} {lesion_name} {method_name}')
        fig.savefig(join(ns.save_dir,
                         f'{img_id}-{lesion_name.replace(" ","_")}-{method_name.replace(" ","_")}.png'))
        if ns.show_plot:
            plt.show()



def bap():
    p = ap.ArgumentParser()
    p.add_argument('--img-ids', nargs='*', help='By default, analyze all imgs.  If this parameter is given, only analyze given imgs:  --img-ids IDRiD_01 IMDiD_02')
    p.add_argument('--labels', nargs='*', default=('MA', 'HE', 'EX', 'SE', 'OD'), help='Lesions to analyze', choices=('MA', 'HE', 'EX', 'SE', 'OD'))
    p.add_argument('--methods', nargs='*', default=tuple(competing_methods.all_methods.keys()), choices=tuple(competing_methods.all_methods.keys()), help='list of methods named in competing_methods.all_methods')
    p.add_argument('--data-dir', default='./data/IDRiD_segmentation', help='Location of the IDRiD dataset')
    p.add_argument('--save-dir', default='./data/histograms_idrid', help='Location of the IDRiD dataset')
    p.add_argument('--show-plot', action='store_true', help='by default, dont show plot.  just save a figure to disk')
    return p


if __name__ == "__main__":
    # use like this
    # python  gen_histograms.py --labels MA HE --methods "Unmodified Image" "Sharpen, t=0.15" --img-ids IDRiD_01 IDRiD_02
    import argparse as ap
    NS = bap().parse_args()
    print(NS)
    main(NS)
