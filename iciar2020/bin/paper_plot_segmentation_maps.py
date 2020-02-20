from matplotlib import pyplot as plt
import pickle
import numpy as np
import os
import torch
from model_configs import getitem_transforms
from dataclasses import dataclass
from argparse_dataclass import ArgumentParser
import simplepytorch.datasets as D
from model_configs.shared_preprocessing import preprocess


def load_model(checkpoint_fp, device):
    ch = torch.load(checkpoint_fp, map_location=device)
    mdl = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=4, init_features=128, pretrained=False)
    mdl.load_state_dict(ch['model_state_dict'])
    mdl.to(device)
    mdl.eval()
    return mdl


def yield_imgs(mdl, ns):
    # iterate through images and plot their segmentation maps.
    for n, tensor_np in enumerate(ns.dset):
        X_i, _ = preprocess(tensor_np, 'identity', rot=0, flip_x=False, flip_y=False)
        X_m, ground_truth = preprocess(tensor_np, ns.ietk_method_name, rot=0, flip_x=False, flip_y=False)
        with torch.no_grad():
            yhat = mdl(torch.stack([X_i, X_m]).to(ns.device)).cpu().numpy()

        identity_result = yhat[0]
        enhanced_result = yhat[1]

        assert yhat.shape[1] == 4
        ground_truth = ground_truth.bool().numpy()
        assert ground_truth.shape[0] == 4
        yield X_i, X_m, enhanced_result, identity_result, ground_truth


def plot_rite_segmentation(X_i, X_m, enhanced_result, identity_result,
                           ground_truth, ns):
    ARTERY, OVERLAP, VEIN, VESSEL = [0,1,2,3]
    y = ground_truth

    figs = {}
    for name, input_img, yh in [(ns.ietk_method_name, X_m, enhanced_result),
                                ('identity', X_i, identity_result)]:
        yh = yh > 0.5
        y = y > 0.5
        red=yh[ARTERY] | yh[OVERLAP] | yh[VEIN] | yh[VESSEL]
        #  green=yh[ARTERY] | yh[OVERLAP] | yh[VEIN] | yh[VESSEL]
        green=y[ARTERY] | y[OVERLAP] | y[VEIN] | y[VESSEL]
        blue=yh[ARTERY] | yh[OVERLAP] | yh[VEIN] | yh[VESSEL]
        z = np.dstack([red, green, blue])*255
        #  import IPython; IPython.embed()
        #  import sys ; sys.exit()
        #  fig, axs = plt.subplots(1, 3)
        #  axs[0].imshow(yh[ARTERY])
        #  axs[1].imshow(yh[VEIN])
        #  axs[2].imshow(yh[VESSEL])
        #  print(yh.max(), yh.min())

        # --> secondary colors are good
        # yellow: artery agreement (true positive)
        # cyan: vein agreement (true positive)
        # white: vessel agreement. (true positive)  the ground truth was uncertain and labeled as both artery and vein.
        # --> primary colors are bad
        # green: ground truth that model missed (false negative)
        # red or blue: artery or vein not in ground truth (false positive)
        fig, axs = plt.subplots(1, 3, num=1, figsize=(10,10))
        axs[0].imshow(np.dstack([red, green, blue])*255)
        axs[0].set_title('Segmentation')
        axs[1].imshow(X_i.permute(1,2,0).numpy())
        axs[1].set_title('Enhanced Image (%s)' % ns.ietk_method_name)
        axs[2].imshow(X_m.permute(1,2,0).numpy())
        axs[2].set_title('Unmodified Image')
        figs[name] = fig
    return figs


@dataclass
class params:
    device: str = 'cuda'
    ietk_method_name: str = 'sC+sX'  # TODO: best one
    img_save_dir: str = './data/plots/rite'
    checkpoint_fp: str = './data/results/R2.2-{ietk_method_name}/model_checkpoints/epoch_best.pth'

    #  get dataset
    dset = D.RITE(
        use_train_set=False,
        getitem_transform=D.RITE.as_tensor(['av', 'vessel'], return_numpy_array=True))

    n_imgs: int = 10


def main(ns: params):
    os.makedirs(ns.img_save_dir, exist_ok=True)
    mdl = load_model(ns.checkpoint_fp.format(**ns.__dict__), ns.device)
    for n, tup in enumerate(yield_imgs(mdl, ns)):
        figs = plot_rite_segmentation(*tup, ns)
        for k, fig in figs.items():
            save_fp = f'{ns.img_save_dir}/rite-seg-{n}-{k}.png'
            fig.savefig(save_fp, bbox_inches='tight')
        # iterate through images and show correlation?  can we just train model on test set?
            #  perf = (yhat > 0.5 == y).sum((1,2,3))
            #  print(perf)  # vector of 2 scalars
            # should see  perf[0] <= perf[1]
        if n >= ns.n_imgs:
            break


if __name__ == "__main__":
    ns = ArgumentParser(params).parse_args()
    print(ns)
    main(ns)
