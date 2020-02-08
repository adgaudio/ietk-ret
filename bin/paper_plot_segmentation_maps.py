from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch
from model_configs import getitem_transforms
import screendr.datasets as D
from dataclasses import dataclass
from model_configs.shared_preprocessing import preprocess
from argparse_dataclass import ArgumentParser


def load_model(checkpoint_fp, device):
    with open(checkpoint_fp, 'rb') as fin:
        ch = pickle.load(fin)
    mdl = torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 'unet',
        in_channels=3, out_channels=4, init_features=128, pretrained=False)
    mdl.load_state_dict(ch['model_state_dict'])
    mdl.to(device)
    mdl.eval()
    return mdl


def yield_imgs(mdl, ns):
    # iterate through images and plot their segmentation maps.
    for n, (X, ground_truth) in enumerate(ns.dset):
        X_i = preprocess(X, 'identity', rot=0, flip_x=False, flip_y=False)
        X_m = preprocess(X, ns.ietk_method_name, rot=0, flip_x=False, flip_y=False)
        yhat = ns.model(torch.stack([X_i, X_m]).to(ns.device)).cpu().numpy()

        identity_result = yhat[0]
        enhanced_result = yhat[1]

        assert yhat.shape[1] == 4
        assert ground_truth.shape[2] == 4
        yield X_i, X_m, enhanced_result, identity_result, ground_truth


def plot_rite_segmentation(X_i, X_m, enhanced_result, identity_result,
                           ground_truth, image_fname, ns):
    ARTERY, _, VEIN, _ = [0,1,2,3]
    y = ground_truth

    for name, input_img, yh in [(ns.ietk_method_name, X_m, enhanced_result),
                                ('identity', X_i, identity_result)]:
        red=yh[ARTERY]
        green=y[ARTERY] | y[VEIN]
        blue=yh[VEIN]
        # --> secondary colors are good
        # yellow: artery agreement (true positive)
        # cyan: vein agreement (true positive)
        # white: vessel agreement. (true positive)  the ground truth was uncertain and labeled as both artery and vein.
        # --> primary colors are bad
        # green: ground truth that model missed (false negative)
        # red or blue: artery or vein not in ground truth (false positive)
        f, axs = plt.subplots(1, 3, num=1)
        axs[0].imshow(np.dstack([red, green, blue]))
        axs[0].set_title('Segmentation')
        axs[1].imshow(X_i.permute(1,2,0).numpy())
        axs[1].set_title('Enhanced Image (%s)' % ns.ietk_method_name)
        axs[2].imshow(enhanced_result)
        axs[2].set_title('Unmodified Image')
    return fig


@dataclass
class params:
    device: str = 'cuda'
    ietk_method_name: str = 'A+C'  # TODO: best one
    img_save_dir: str = './data/plots/rite'
    checkpoint_fp: str = './data/results/R2.2-{ietk_method_name}/model_checkpoints/best_epoch.pth'

    #  get dataset
    dset = D.RITE(use_train_set=False, getitem_transform=getitem_transforms(
        'test', 'identity', D.RITE.as_tensor(return_numpy_array=True)))

    n_imgs: int = 10


def main(ns: params):
    mdl = load_model(ns.checkpoint_fp.format(**ns.__dict__), ns.device)
    for n, tup in enumerate(yield_imgs(mdl, ns)):
        image_fname = f'{ns.img_save_dir}/rite-seg-{ns.ietk_method_name}-{n}.png'
        fig = plot_rite_segmentation(*tup, image_fname, ns)
        fig.savefig(image_fname, bbox_inches='tight')
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
