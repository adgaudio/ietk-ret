from matplotlib import pyplot as plt
from dataclasses import dataclass, field
from argparse_dataclass import ArgumentParser
import numpy as np
from typing import List
import os
from os.path import basename
import screendr.datasets as D
import model_configs.shared_preprocessing as SP


@dataclass
class params:
    dset_qualdr = D.QualDR(img_transform=np.array, getitem_transform=None)
    methods: List[str] = field(default_factory=lambda: ['A', 'B', 'X'])
    save_fig_dir: str = './data/plots/qualitative'
    num_imgs: int = 10


def save_plot(ns, img_idx):
    fig, axs = plt.subplots(2, len(ns.methods))
    axs[0,0].set_ylabel('Enhanced')
    axs[1,0].set_ylabel('Original')
    [ax.axis('off') for ax in axs.ravel()]
    for ax_idx, ietk_method_name in enumerate(ns.methods):
        dct = ns.dset_qualdr[img_idx]
        img = np.array(dct['image'])
        enhanced_img, _ = SP.preprocess(
            img, ietk_method_name, rot=0, flip_y=False, flip_x=False,
            resize_to=img.shape[:2], crop_to_size=img.shape[:2])
        enhanced_img = (enhanced_img.permute(1,2,0).numpy() * 255).astype('uint8')

        axs[0, ax_idx].imshow(enhanced_img)
        axs[0, ax_idx].set_title(ietk_method_name)
        axs[1, ax_idx].imshow(img)
    fig.tight_layout()
    fig.savefig(f'{ns.save_fig_dir}/qualitative-{basename(dct["fp"])}')
    return fig


def main(ns):
    os.makedirs(ns.save_fig_dir, exist_ok=True)

    img_idxs = np.random.randint(0, len(ns.dset_qualdr), ns.num_imgs)
    for img_idx in img_idxs:
        save_plot(ns, img_idx)


if __name__ == "__main__":
    main(ArgumentParser(params).parse_args())
