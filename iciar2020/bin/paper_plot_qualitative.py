from matplotlib import pyplot as plt
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
import numpy as np
from typing import List
import os
from os.path import basename
import simplepytorch.datasets as D
import model_configs.shared_preprocessing as SP

@dataclass
class params:
    dset_qualdr = D.QualDR(img_transform=np.array, getitem_transform=None)
    methods: List[str] = field(default_factory=lambda: [
        #  'identity', 'A+B+X', 'sA+sB+sC+sW+sX', 'sA+sC+sX+sZ', 'sA+sC+sX', 'A+B+C+X', 'sC+sX'])
        'identity', 'sA+sC+sX+sZ', 'sA+sB+sC+sW+sX', 'A+B+C+X', 'sC+sX'])
        #  'identity', 'identity'])
    save_fig_dir: str = './data/plots/qualitative'
    num_imgs: int = 100
    img_idx: int = None
    overwrite: int = True


def save_plot(ns, img_idx):
    for ax_idx, ietk_method_name in enumerate(ns.methods):
        fig, axs = plt.subplots(1, len(ns.methods), num=1, figsize=(4*len(ns.methods), 4))
        [ax.cla() for ax in axs.ravel()]
        [ax.axis('off') for ax in axs.ravel()]
        dct = ns.dset_qualdr[img_idx]
        save_fp = f'{ns.save_fig_dir}/qualitative-{img_idx}-{basename(dct["fp"])}'
        if ns.overwrite is False and os.path.exists(save_fp):
            print('skip.  file already exists.', save_fp)
            return
        img = np.array(dct['image'])
        enhanced_img, _ = SP.preprocess(
            img, ietk_method_name, rot=0, flip_y=False, flip_x=False,
            resize_to=img.shape[:2], crop_to_size=img.shape[:2])
        enhanced_img = (enhanced_img.permute(1,2,0).numpy() * 255).astype('uint8')

        axs[ax_idx].imshow(enhanced_img)
        axs[ax_idx].set_title(ietk_method_name, fontsize=20)
        axs[ax_idx].axis('off')
    fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9, bottom=0.1)
    #  fig.tight_layout()
    fig.savefig(save_fp, bbox_inches='tight')
    return fig


def main(ns):
    os.makedirs(ns.save_fig_dir, exist_ok=True)

    if ns.img_idx is not None:
        img_idxs = [ns.img_idx]
    else:
        img_idxs = np.random.randint(0, len(ns.dset_qualdr), ns.num_imgs)
    for img_idx in img_idxs:
        save_plot(ns, img_idx)
        #  plt.show(block=False)
        #  plt.pause(0)


if __name__ == "__main__":
    main(ArgumentParser(params).parse_args())
