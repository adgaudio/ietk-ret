import cv2
import logging
import numpy as np
import os
from os.path import join
import torch.nn
import torch.utils.data as TD
import torchvision.transforms as tvt
from functools import partial

from simplepytorch import api
from simplepytorch import datasets as D

import ietk.util
import ietk.methods
from model_configs.qualdr_grading import eval_perf
from model_configs.shared_preprocessing import preprocess

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def get_train_val_idxs(dset_size, train_frac):
    """Generate index values for the training and validation sets."""
    tr_size = int(dset_size * train_frac)
    idx_shuffled = torch.randperm(dset_size).tolist()
    return idx_shuffled[:tr_size], idx_shuffled[tr_size:]


def getitem_transforms(train_val_test: str, ietk_method_name: str,
                       initial_transform):
    """For RITE and IDRiD datasets"""
    assert train_val_test in ['train', 'val', 'test']
    fns = [initial_transform]
    if train_val_test == 'train':
        fns.append(partial(preprocess, method_name=ietk_method_name))
    else:
        fns.append(partial(preprocess, method_name=ietk_method_name,
                           rot=0, flip_y=False, flip_x=False))
    return tvt.Compose(fns)


def IDRiD_Segmentation_PREPROCESSED(base_dir):
    """Load the pickled dataset with a different base directory"""
    def _IDRiD_Segmentation_wrapper(use_train_set, *, getitem_transform, **kwargs):
        train_or_test = 'train' if use_train_set else 'test'
        return D.PickledDicts(
            base_dir.format(train_or_test=train_or_test),
            getitem_transform=lambda dct: getitem_transform(dct['tensor']),
            **kwargs)
    return _IDRiD_Segmentation_wrapper


class BDSSegment(api.FeedForwardModelConfig):
    epochs = 100
    batch_size = 2  # TODO
    run_id = 'test'

    def __init__(self, cfg):
        data_name = cfg.get('data_name') or self.data_name
        if data_name == 'rite':
            self._num_output_channels = 4
        elif data_name == 'idrid':
            self._num_output_channels = 5
        else:
            raise NotImplementedError()
        super().__init__(cfg)
        self._early_stopping = api.EarlyStopping(['MCC_val', 'Dice_val'], 30)
        os.makedirs(self.checkpoint_img_dir.format(self=self), exist_ok=True)

    __model_params = api.CmdlineOptions(
        'model', {'name': 'unused',  # unused
                  'num_classes': -1  # unused
                  })

    def get_model(self):
        if self.model_name != 'unused':
            return super().get_model()
        else:
            model = torch.hub.load(
                'mateuszbuda/brain-segmentation-pytorch', 'unet',
                in_channels=3, out_channels=self._num_output_channels,
                init_features=128, pretrained=False)
            return model.to(self.device)

    def get_lossfn(self):
        if self.data_name == 'idrid':
            w = D.IDRiD_Segmentation.POSITIVE_PIXELS_PER_CATEGORY_TRAIN
        elif self.data_name == 'rite':
            w = torch.tensor([1.,1.,1.,1.])
        else:
            raise NotImplementedError()
        w = w / w.sum()
        per_channel_weights = (w.max() / w).to(self.device)
        def loss_cross_entropy_per_output_channel(input, target):
            batch_size = target.shape[0]
            #  return torch.nn.functional.binary_cross_entropy(
                #  input.view(batch_size, -1), target.view(batch_size, -1))

            assert self._num_output_channels == target.shape[1]
            losses = torch.stack([
                torch.nn.functional.binary_cross_entropy(
                    input[:,x].view(batch_size, -1), target[:,x].view(batch_size, -1),
                    reduction='sum')
                for x in range(target.shape[1])])
            losses = losses * per_channel_weights
            return losses.sum()
        return loss_cross_entropy_per_output_channel

    __optimizer_params = api.CmdlineOptions(
        'optimizer', {
            'lr': 0.001, 'weight_decay': 0.0001, 'betas': (0.9, 0.999) },)

    def get_optimizer(self):
        return torch.optim.Adam(
            self.model.parameters(), **self.__optimizer_params.kwargs(self))

    __data_params = api.CmdlineOptions(
        'data', {'name': 'rite',  # choices: idrid, rite
                 'use_train_set': True,  # otherwise use test set
                 'train_val_split': 0.8,  # use random 20% of train set as validation
                 'idrid_base_dir': './data/preprocessed/idrid-identity/{train_or_test}',
                 })
    __ietk_params = api.CmdlineOptions(
        # which preprocessing method to use
        'ietk', {'method_name': str},
        choices={'method_name': ietk.methods.all_methods})

    def _get_datasets(self, name: str, dataset_kls, initial_getitem_transform):
        args = (self.ietk_method_name, initial_getitem_transform)
        dset_train = dataset_kls(
            use_train_set=True, img_transform=None,
            getitem_transform=getitem_transforms('train', *args))
        dset_val = dataset_kls(
            use_train_set=True, img_transform=None,
            getitem_transform=getitem_transforms('val', *args))
        dset_test = dataset_kls(
            use_train_set=False, img_transform=None,
            getitem_transform=getitem_transforms('test', *args))
        dset_tr_idxs, dset_val_idxs = get_train_val_idxs(
            len(dset_train), self.data_train_val_split)
        return {
            f'{name}_train': TD.Subset(dset_train, dset_tr_idxs),
            f'{name}_val': TD.Subset(dset_val, dset_val_idxs),
            f'{name}_test': dset_test, }

    def get_datasets(self):
        dsets = {}
        dsets.update(self._get_datasets(
            'idrid', IDRiD_Segmentation_PREPROCESSED(self.data_idrid_base_dir),
            lambda x: x))  # D.IDRiD_Segmentation.as_tensor(return_numpy_array=True)))
        dsets.update(self._get_datasets(
            'rite', D.RITE, D.RITE.as_tensor(['av', 'vessel'], return_numpy_array=True)))
        return super().get_datasets(dsets)

    def get_category_names(self):
        if self.data_name == 'rite':
            return D.RITE.LABELS_AV + ['vessels']
        elif self.data_name == 'idrid':
            return D.IDRiD_Segmentation.LABELS

    if torch.cuda.device_count():
        data_loader_num_workers = int(torch.multiprocessing.cpu_count()/torch.cuda.device_count()-3)
    else:
        data_loader_num_workers = int(torch.multiprocessing.cpu_count())-3

    def get_data_loaders(self):
        kws = dict(
            batch_size=self.batch_size,
            pin_memory=True if self.data_name in ['rite'] else False,
            num_workers=self.data_loader_num_workers,)
        if not self.data_use_train_set:
            ldct = {
                'test': torch.utils.data.DataLoader(
                    getattr(self.datasets, f'{self.data_name}_test'), **kws),
            }
        else:
            ldct = {
                'train': torch.utils.data.DataLoader(
                    getattr(self.datasets, f'{self.data_name}_train'), shuffle=True, **kws),
                'val': torch.utils.data.DataLoader(
                    getattr(self.datasets, f'{self.data_name}_val'), **kws),
            }
        return super().get_data_loaders(ldct)

    def eval_early_stopping(self):
        return self._early_stopping.should_stop(self.cur_epoch)

    def log_minibatch(self, batch_idx, X, y, yhat, loss, dloader='train'):
        cache = self.epoch_cache.setdefault(dloader, api.Cache())

        # --> compute confusion matrix
        # columns are predicted.
        # rows are known true
        # top left is true positive
        yhat_th = yhat > 0.5
        y_bool = y.bool()

        def compute_cm(yh, y):
            a,b,c = (yh & y).sum(), yh.sum(), y.sum()
            a,b,c = [x.cpu() for x in (a,b,c)]
            tn = (~yh & ~y).sum().cpu()  # true negatives
            cm = torch.tensor([[a, c-a], [b-a, tn]])
            return cm
        cm = compute_cm(yhat_th, y_bool)
        cache.add('confusion_matrix', cm.cpu().float())

        # also log the confusion matrix for each task category (ie IDRiD lesion)
        for i, category in enumerate(self.get_category_names()):
            cm = compute_cm(yhat_th[:, i], y_bool[:, i])
            cache.add(f'confusion_matrix_{category}', cm.cpu().float())

        # --> update loss
        cache.streaming_mean(f'loss', loss.item(), y.shape[0])

    def log_epoch(self, dloader='train'):
        cache = self.epoch_cache[dloader]
        eld = {}

        (tp, fn), (fp, tn) = cache['confusion_matrix']
        # --> compute dice
        eld['Dice'] = (2*tp / (2*tp + fp + fn)).item()
        # --> mcc
        eld['MCC'] = api.confusion_matrix_stats.matthews_correlation_coeff(
            cache['confusion_matrix']).item()
        eld['Loss'] = cache['loss'].mean

        # --> compute dice for each category
        for category in self.get_category_names():
            (tp, fn), (fp, tn) = cache[f'confusion_matrix_{category}']
            eld[f'Dice_{category}'] = (2*tp / (2*tp + fp + fn)).item()
            eld[f'MCC_{category}'] = api.confusion_matrix_stats.matthews_correlation_coeff(
                cache[f'confusion_matrix_{category}']).item()

        # --> add suffix to all keys
        eld = {f'{k}_{dloader}': v for k, v in eld.items()}

        if dloader in {'train', 'test'}:
            if dloader == 'train':  # append val cache too
                if self.data_train_val_split == 1:
                    eld.update({k.replace('_train', '_val'): 0 for k in eld})
                else:
                    eval_perf(self, 'val')
                    eld.update(self.log_epoch('val'))
            super().log_epoch(eld)
        else:
            assert dloader == 'val', 'sanity check'
            # store whether this was best epoch
            self._early_stopping.is_best_performing_epoch(
                self.cur_epoch, {k: eld[k] for k in {'Dice_val', 'MCC_val'}})
            return eld  # for recursion

    def get_log_header(self):
        fields = ['Dice', 'MCC', 'Loss']
        fields.extend([f'{f}_{cat}' for f in ['Dice', 'MCC']
                       for cat in self.get_category_names()])
        return super().get_log_header([
            f'{metric}_{dloader}'
            for metric in fields
            for dloader in (['train', 'val']
                            if self.data_use_train_set else ['test'])])

    __checkpoint_params = api.CmdlineOptions(
        'checkpoint', {
            'fname': 'epoch_best.pth',
            'img_dir': '{self.base_dir}/results/{self.run_id}/images/'})

    def save_checkpoint(self):
        if self._early_stopping.is_best_performing_epoch(self.cur_epoch):
            super().save_checkpoint(force_save=True)
            # save image (during training)
            for X, y in self.data_loaders.val:
                yhat = self.model(X.to(self.device))
                self._save_imgs_to_file(X, yhat)
                break

    def _save_imgs_to_file(self, X, yhat, start_idx=0):
        """Save a minibatch output from model to disk"""
        yhat = yhat.cpu().numpy()
        X = X.cpu().permute(0,2,3,1).numpy()
        for i in range(yhat.shape[0]):
            # --> write input image
            fp = join(
                self.checkpoint_img_dir,
                f'{self.cur_epoch}-{start_idx+i}-input.tiff'
            ).format(self=self, i=i)
            was_written = cv2.imwrite(fp, X[i][:,:,[2,1,0]])
            assert was_written
            # --> write output segmentation images
            for ch, cat in enumerate(self.get_category_names()):
                fp = join(
                    self.checkpoint_img_dir,
                    f'{self.cur_epoch}-{start_idx+i}-{cat}.tiff'
                ).format(self=self, i=i)
                was_written = cv2.imwrite(
                    fp, ((yhat[i,ch]>0.5)*255).astype('uint8'))
                assert was_written

    def get_checkpoint_state(self):
        dct = super().get_checkpoint_state()
        dct.update({
            '_early_stopping': self._early_stopping
        })
        return dct

    def load_checkpoint(self):
        checkpoint = super().load_checkpoint()
        if checkpoint:
            self._early_stopping = checkpoint['_early_stopping']

    debug_visualize_preprocessing = False

    def run(self):
        if self.debug_visualize_preprocessing:
            from matplotlib import pyplot as plt
            plt.ion()
            tt = 'train' if self.data_use_train_set else 'test'
            for x,y in self.datasets._asdict()[f'{self.data_name}_{tt}']:
                x = x.permute(1,2,0).numpy()
                plt.imshow(x) ; #plt.pause(0.4)
                plt.gcf().suptitle(self.ietk_method_name)
                plt.waitforbuttonpress()
            import IPython ; IPython.embed() ; import sys ; sys.exit()
        elif self.data_use_train_set:
            self.train()
        else:
            self.model.eval()
            with torch.no_grad():
                self.epoch_cache.clear()
                eval_perf(self, 'test')
                self.log_epoch('test')
                # --> plot up to 30 images and predictions in the dataset.
                N = 0
                for X, y in self.data_loaders.test:
                    if N > 30: break
                    yhat = self.model(X.to(self.device))
                    self._save_imgs_to_file(X, yhat, N)
                    N += X.shape[0]


if __name__ == "__main__":
    #  main()
    from matplotlib import pyplot as plt


    def testing():
        dset = D.RITE(img_transform=None, getitem_transform=getitem_transforms(
            'val', 'identity',
            D.RITE.as_tensor(return_numpy_array=True)))
        #  dset2 = D.RITE(img_transform=None, getitem_transform=None)

        #  dset = D.IDRiD_Segmentation(img_transform=None, getitem_transform=getitem_transforms(
            #  'val', 'identity',
            #  D.IDRiD_Segmentation.as_tensor(return_numpy_array=True)))
        #  dset2 = D.IDRiD_Segmentation(img_transform=None, getitem_transform=None)

        f, axs = plt.subplots(2, 3)
        for id in torch.utils.data.RandomSampler(dset):
            #  dct = dset2[id]
            #  im, mask = (
                #  torch.tensor(np.array(dct['image'])).permute(2,0,1),
                #  torch.tensor(np.array(dct['av'])).permute(2,0,1))
            #  imnp, masknp = np.array(dct['image'])/255, np.array(dct['av'])
            #  bg = ietk.util.get_background(imnp)
            #  z=ietk.methods.sharpen_img.sharpen(imnp, bg)
            #  z=ietk.methods.all_methods['A+B'](imnp, ~bg)
            #  plt.figure(1) ; plt.imshow(z) ; plt.pause(0.1)
            #  continue

            im, mask = dset[id]
            _im = im.permute(1,2,0).numpy()
            _mask = mask.permute(1,2,0).numpy()
            print(_mask.min(), _mask.max())
            stack = np.dstack([_im*255, _mask])
            im2, l2 = preprocess(stack, 'C')
            im3, l3 = preprocess(stack, 'A+B+C+W+X', rot=0, flip_y=False, flip_x=False)
            print('max', im.max(), im2.max(), im3.max(), l3.max())
            print('min', im.min(), im2.min(), im3.min(), l3.min())

            for ax, I in zip(axs.ravel(), [im, im2, im3, mask, l2, l3]):
                ax.imshow(I.permute(1,2,0).numpy())
            plt.pause(.1)
    testing()
