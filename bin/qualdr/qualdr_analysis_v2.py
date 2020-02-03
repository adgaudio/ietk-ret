# find list of images that the identity model gets wrong.
# visualize GradCAM for that image.
# take the best performing image and visualize gradcam for it too.

#  (one visual per category)
from efficientnet_pytorch import EfficientNet
from matplotlib import pyplot as plt
import numpy as np
import re
import torch

import model_configs.qualdr_grading as IC
import screendr.datasets as D


def parse_method_name(run_id):
    return run_id.split('-', 1)[1]


def load_model(checkpoint_fp, device):
    model_name = 'efficientnet-b4'
    model = EfficientNet.from_pretrained(model_name, num_classes=13)
    checkpoint = torch.load(checkpoint_fp, map_location=device)
    model = model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    return model


def get_difficult_imgs(checkpoint_fps, dsets, device):
    # TODO: test all of this.
    failed_img_fps = []
    poor_quality_img_fps = []
    # --> load identity model
    model_identity = load_model(checkpoint_fps['identity'])
    for X, y, fp in torch.utils.data.RandomSampler(
            dsets['identity_for_finding_failures']):
        X.to(device)
        y.to(device)
        yhat = model_identity(X).squeeze()
        # --> was the model wrong?
        for k, (s, e) in D.QualDR_Grading.GRADES_FLAT_IDX_RANGES:
            yh = yhat[:, s:e]
            #  _onehot = torch.zeros_like(yh)
            #  _onehot[torch.arange(e-s).repeat(batch_size, 1) == yh.argmax(1).reshape(-1, 1)] = 1
            #  yh = _onehot
            model_was_wrong = yh.argmax() != y.argmax()
            if model_was_wrong:
                failed_img_fps.append((k, fp))
            if y[e-1] == 1:
                poor_quality_img_fps.append((k, fp))
            import IPython ; IPython.embed() ; import sys ; sys.exit()
    return {'failed': failed_img_fps, 'poor_quality': poor_quality_img_fps}


def preprocess_image(img, cat):
    """cat is in {identity,retinopathy,maculopathy,photocoagulation}"""
    method_name = parse_method_name(run_ids[cat])
    im2 = IC.preprocess(img, method_name=method_name).clip(0, 1)
    # caused by the guided filter blur function in sharpen of ietk when there
    # are large values.
    im2[np.isnan(im2)] = 0
    return im2


N = 4  # num images to gradcam-ize
device = 'cuda:1'
run_ids = {
    'identity': 'Q1-identity',
    'retinopathy': 'Q1-A',
    'maculopathy': 'Q1-sA+sB+sC+sW+sX',
    'photocoagulation': 'Q1-B+X',
}

# checkpoints
_checkpoint_fp = 'data/results/{run_id}/model_checkpoints/epoch_best.pth'
checkpoint_fps = {
    k: _checkpoint_fp.format(run_id=run_id) for k, run_id in run_ids.items()}

# dsets.
dsets = {
    'identity_for_finding_failures': D.PickledDicts(
        f'./data/preprocessed/arsn_qualdr-ietk-{parse_method_name(run_ids["identity"])}/test',
        img_transform=IC.qualdr_preprocessed_img_transform(
            IC.BDSQualDR, 'test'),
        getitem_transform=lambda x: (
                x['image'],
                D.QualDR_Grading.get_diagnosis_grades(
                    x['json']),
            x['fp']),
                    ),
    'qualdr_for_gradcam': D.QualDR(
                        default_set='test',
                        img_transform=None,
        getitem_transform=None)}

# todo: follow this
# https://github.com/kazuto1011/grad-cam-pytorch/blob/master/main.py

# 1. determine which images the identity model got wrong and store a list of fps.
# 2. for a subset of these imgs, apply gradcam for each of the four model

# find which images the identity messes up on
difficult_img_fps = get_difficult_imgs(checkpoint_fps, dsets, device)

import sys ; sys.exit()

# evaluate gradcam on each of the models and failed imgs.
for fp in failed_img_fps[:N]:
    # load img.
    orig_img = plt.imread(fp)
    for cat in run_ids:
        model = load_model(cat)
        img = preprocess_image(orig_img, cat)
        img.to(device)
        # evaluate gradcam
        #  yhat = model(img)
        #  TODO ... GradCAM and save result.

        # TODO?: evaluate whether identity model improves with preprocessing
        # (why should I do this?
        if cat != 'identity':
            yhat = model_identity(orig_img)
            yhat = model_identity(img)
            ...  # maybe visualize differences in distribution.  does it shift?
            # or maybe more appropriate to just evaluate test performance on a
            # different pickled dict test set.

# do the preprocessed models find features that identity model doesn't?  Are they learning to "dumb down" their results?
    # - three identity gradcam imgs (top row), three preprocessed gradcam outputs (bottom row).
    # look at failed images
    # look at poor quality (RX,MX,PX) images.
# does the identity model perform better with the preprocessing (ie no fine-tuning)?
    # this suggests that the pre-processing by itself makes the task easier
    # rather than harder.  (this is useful since we can't assume from looking
    # at difference in performance that the preprocessing makes task easier -
    # it might just be adding regularization)
