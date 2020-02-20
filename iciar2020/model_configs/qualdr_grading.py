from collections import namedtuple
from functools import partial
import logging
#  import cupy as np
import numpy as np
import pandas as pd
import time
import torch.nn
import torchvision.transforms as tvt

from simplepytorch import api
from simplepytorch import datasets as D
from os.path import join
from model_configs.shared_preprocessing import preprocess, cutout_inplace

import ietk.util
import ietk.methods

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


BestPerf = namedtuple('BestPerf', ['value', 'epoch'])


class BDSQualDR(api.FeedForwardModelConfig):
    """Evaluates the Brighten Darken Sharpen pre-processing methods, black box style"""
    epochs = 60
    batch_size = 8  #20 b0   #10 b3  # 7 b4
    run_id = 'test'

    __checkpoint_params = api.CmdlineOptions(
        'checkpoint', {
            'interval': 5,  # --checkpoint-interval 1
        })

    log_minibatch_interval = 1  # need this to record data.


    def __init__(self, config_override_dict):
        self.train_or_test_str = 'train' if config_override_dict['data_use_train_set'] else 'test'
        super().__init__(config_override_dict)

        self.logger_hdf = api.LogRotate(api.HDFLogger)(
            join(self.base_dir, 'results', self.run_id, f'perf_matrices_{self.train_or_test_str}.h5'),
            header=self.get_loghdf_header())

    debug_visualize_preprocessing = False

    def run(self):
        if self.debug_visualize_preprocessing:
            from matplotlib import pyplot as plt
            plt.ion()
            tt = 'train' if self.data_use_train_set else 'test'
            for x,y in self.data_loaders._asdict()[f'{tt}']:
                for i in range(x.shape[0]):
                    plt.imshow(x[i].permute(1,2,0).numpy()) ; #plt.pause(0.4)
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

    def get_loghdf_header(self):
        h = [
            f'{prefix}_{category}_{train_val_test}'
            for category in ['retinopathy', 'maculopathy', 'photocoagulation']
            for prefix in ['CM_soft', 'CM_hard']
            for train_val_test in (['train', 'val'] if self.data_use_train_set else ['test'])]
        h.append(f'perf_{"train" if self.data_use_train_set else "test"}_csv')
        return h

    def get_log_header(self):
        return super().get_log_header([
            f'{metric}_{cat}_{dloader}'
            for metric in ['acc', 'loss', 'MCC_soft', 'MCC_hard']
            for cat in D.QualDR_Grading.GRADE_CATEGORIES
            for dloader in (['train', 'val'] if self.data_use_train_set else ['test'])])

    def get_log_filepath(self):
        return f'{self.base_dir}/results/{self.run_id}/perf_{self.train_or_test_str}.csv'

    def get_lossfn(self):
        return D.QualDR_Grading.loss_cross_entropy_per_category

    __optimizer_params = api.CmdlineOptions(
        'optimizer', {'name': 'adam', 'lr': 0.0002, 'weight_decay': 0.0001},
        #  'optimizer', {'name': 'rmsprop', 'lr': 0.256, 'weight_decay': 1e-5, momentum: 0.9, },
        choices={'name': ['adam']}
    )

    def get_optimizer(self):
        if self.optimizer_name == 'adam':
            kws = self.__optimizer_params.kwargs(self) ; kws.pop('name')
            return torch.optim.Adam(self.model.parameters(), **kws)
        else:
            raise NotImplementedError()

    __data_params = api.CmdlineOptions(
        'data', {'name': 'qualdr',  # choices: qualdr, (maybe messidor soon)
                 'use_train_set': True,  # otherwise use test set
                 'train_val_split': 0.8,  # use random 20% of train set as validation
                 })

    __ietk_params = api.CmdlineOptions(
        # which preprocessing method to use
        'ietk', {'method_name': str},
        choices={'method_name': ietk.methods.all_methods})
    preprocess_clip_imgs = True  # whether to clip to [0-1] or allow any values.
    preprocess_mul255 = False  # whether to pass [0-1] normalized images or [0-255] (after clipping)

    def get_datasets(self):
        return super().get_datasets({
            #  'qualdr': D.QualDR(
                #  default_set=self.train_or_test_str,
                #  img_transform=qualdr_img_transform(self.ietk_method_name),),
            'qualdr': D.PickledDicts(
                f'./data/preprocessed/arsn_qualdr-ietk-identity/{self.train_or_test_str}',
                img_transform=qualdr_preprocessed_img_transform(self, self.train_or_test_str),
                getitem_transform=lambda x: (
                    x['image'], D.QualDR_Grading.get_diagnosis_grades(x['json'])),
                 ),
            'qualdr_no_cutout': D.PickledDicts(
                f'./data/preprocessed/arsn_qualdr-ietk-identity/{self.train_or_test_str}',
                img_transform=qualdr_preprocessed_img_transform(self, 'test'),
                getitem_transform=lambda x: (
                    x['image'], D.QualDR_Grading.get_diagnosis_grades(x['json'])),
                 ),
        })

    debug_small_dataset = 0  # set to num of imgs in the debugging set.
    if torch.cuda.device_count():
        data_loader_num_workers = int(torch.multiprocessing.cpu_count()/torch.cuda.device_count()-3)
    else:
        data_loader_num_workers = int(torch.multiprocessing.cpu_count())-3

    def get_data_loaders(self):
        assert self.data_name == 'qualdr'
        dset = getattr(self.datasets, self.data_name)
        if self.debug_small_dataset:
            _idxs = torch.randperm(len(dset))[:self.debug_small_dataset]
            dset = torch.utils.data.Subset(dset, _idxs)

        kws = dict(batch_size=self.batch_size,
                   pin_memory=True if self.data_name in ['rite'] else False,
                   num_workers=self.data_loader_num_workers,
                   )
        if not self.data_use_train_set:
            return super().get_data_loaders({
                'test': torch.utils.data.DataLoader(dset, **kws)
            })
        else:
            tr_size = int(len(dset) * self.data_train_val_split)
            idx_shuffled = torch.randperm(len(dset)).tolist()
            dtrain = torch.utils.data.Subset(dset, idx_shuffled[:tr_size])
            dval = torch.utils.data.Subset(self.datasets.qualdr_no_cutout, idx_shuffled[tr_size:])
            return super().get_data_loaders({
                'train': torch.utils.data.DataLoader(dtrain, shuffle=True, **kws),
                'val': torch.utils.data.DataLoader(dval, **kws)
                })

    __model_params = api.CmdlineOptions(
        'model', {'name': 'efficientnet-b4',  # --model-name resnet18  (required option)
                  'num_classes': 13,  # --model-num-classes 1000  (required option)
                  })

    def save_checkpoint(self):
        if self.is_best_performing_epoch():
            super().save_checkpoint(force_save=True)

    def get_checkpoint_state(self):
        dct = super().get_checkpoint_state()
        dct.update({
            '_best_perf': self._best_perf
        })
        return dct

    def load_checkpoint(self):
        checkpoint = super().load_checkpoint()
        if checkpoint:
            self._best_perf = checkpoint['_best_perf']

    def eval_early_stopping(self):
        stop = all((self.cur_epoch - perf.epoch) > 15
                   for perf in self._best_perf.values())
        if stop:
            return True
        else:
            return False

    def log_minibatch(self, batch_idx, X, y, yhat, loss, dloader='train'):
        #  super().log_mifnibatch({}, batch_idx=batch_idx)
        y = y.cpu()
        yhat = yhat.cpu()
        if dloader is None:  # "train|test|val"
            dloader = self.train_or_test_str

        # store a confusion matrix per category for hard and soft assignment of
        # predictions to classes.
        CMs_soft = D.QualDR_Grading.create_confusion_matrices(y, yhat, hard_assignment=False)
        CMs_hard = D.QualDR_Grading.create_confusion_matrices(y, yhat, hard_assignment=True)
        for prefix, CMs in [('CM_soft', CMs_soft), ('CM_hard', CMs_hard)]:
            for grade_name, confusion_matrix in CMs.items():
                # store data of form:  {CM_soft_retinopathy_train: val}
                self.epoch_cache.add(
                    f"{prefix}_{grade_name}_{dloader}", confusion_matrix)

        # update loss
        for i, cat in enumerate(D.QualDR_Grading.GRADE_CATEGORIES):
            self.epoch_cache.streaming_mean(f'loss_{cat}_{dloader}', loss[i].item())

    def log_epoch(self, dloader='train'):
        eld = {}

        # first, recursively do logging on the validation set.  (future note: could disable by passing using_val_set=None)
        if dloader == 'train':
            eld_val, hdf_val = self.log_epoch(dloader='val')
            self.is_best_performing_epoch(eld_val)  # store whether this was best epoch
            assert all('val' in k for k in eld_val), 'sanity check'
            assert all('val' in k for k in hdf_val), 'sanity check'
        else: eld_val, hdf_val = {}, {}

        # if not training, may need to collect minibatch cache stats
        if dloader == 'val':  # validation set
            eval_perf(self, dloader)
        cache = self.epoch_cache

        # log CSV data
        for cat in D.QualDR_Grading.GRADE_CATEGORIES:
            # accuracy of form {acc_avged_train: val1, acc_retinopathy_train: val2}
            key = f'acc_{cat}_{dloader}'
            eld[key] = api.confusion_matrix_stats.accuracy(cache[f'CM_hard_{cat}_{dloader}']).item()
            # loss:  {loss_train: val}
            key = f'loss_{cat}_{dloader}'
            eld[key] = cache[key].mean
            # matthews correlation coefficient with hard or soft assignment
            # --> not sure if soft assignment makes sense.  testing it.
            for assignment in ['soft', 'hard']:
                key = f'MCC_{assignment}_{cat}_{dloader}'
                eld[key] = api.confusion_matrix_stats.matthews_correlation_coeff(
                    cache[f'CM_{assignment}_{cat}_{dloader}']).item()

        # log the confusion matrices averaged over epoch to HDF
        def makeframe(cm: np.ndarray):
            df = pd.DataFrame(cm)
            df['epoch'] = self.cur_epoch
            df.set_index('epoch', append=True, inplace=True)
            return df
        hdf = {
            k: makeframe(cache[k].numpy())
            for k in self.get_loghdf_header() if k.endswith(f'_{dloader}')}

        if dloader == 'val':
            return eld, hdf
        else:
            # write the data to file
            # --> update the eld and hdf with validation set data (if any)
            eld.update(eld_val) ; hdf.update(hdf_val)
            # --> extra stuff to ensure the hdf copy of csv perf is in sync
            eld.update({'epoch': self.cur_epoch, 'timestamp': time.time()})
            # --> add all the csv data to hdf store
            hdf[f'perf_{self.train_or_test_str}_csv'] = pd.Series(eld).rename(self.cur_epoch).to_frame().T
            hdf[f'perf_{self.train_or_test_str}_csv'].index.name = 'epoch'

            self.logger_hdf.writerow(hdf)
            super().log_epoch(eld)

    _best_perf = {
        'MCC_hard_retinopathy_val': BestPerf(-2, 0),
        'MCC_hard_maculopathy_val': BestPerf(-2, 0),
        'MCC_hard_photocoagulation_val': BestPerf(-2, 0)}

    def is_best_performing_epoch(self, perf_dct=None):
        """
        Record whether the current epoch is the best one and return True or False.

        If perf_dct is given, update the knowledge of best performing epoch.
        """

        cols = ['MCC_hard_retinopathy_val', 'MCC_hard_maculopathy_val', 'MCC_hard_photocoagulation_val']

        if perf_dct is not None:
            for c in cols:
                cur_perf = BestPerf(perf_dct[c], self.cur_epoch)
                if self._best_perf[c].value < cur_perf.value:
                    self._best_perf[c] = cur_perf
        for c in cols:
            if self.cur_epoch == self._best_perf[c].epoch:
                return True
        return False


def eval_perf(config, dloader:str):
    """Evaluate model performance for an epoch.

    `dloader` - either 'val' or 'test'
    """
    config.model.eval()
    data_loader = config.data_loaders._asdict()[dloader]
    for batch_idx, (X, y) in enumerate(data_loader):
        X, y = X.to(config.device), y.to(config.device)
        yhat = config.model(X)
        loss = config.lossfn(yhat, y)

        config.log_minibatch(batch_idx, X, y, yhat, loss, dloader=dloader)


def clip01(im):
    return im.clip(0, 1)


def checknan(im):
    assert (~torch.isnan(im)).all()
    assert (~torch.isinf(im)).all()
    return im


def qualdr_preprocessed_img_transform(config, train_or_test):
    if train_or_test == 'test':
        # don't apply cutout to test set.
        return tvt.Compose([
            partial(preprocess, method_name=config.ietk_method_name,
                    rot=0, flip_y=False, flip_x=False),
            lambda im_mask: im_mask[0],
            lambda x: (x*255 if config.preprocess_mul255 else x),
            checknan,
        ])
    else:
        return tvt.Compose([
            partial(preprocess, method_name=config.ietk_method_name),
            lambda x: (x*255 if config.preprocess_mul255 else x),
            lambda im_mask: im_mask[0],
            checknan,
            cutout_inplace,
            cutout_inplace,
            cutout_inplace,
        ])


if __name__ == "__main__":
    #  z = BDSQualDR({'data_use_train_set' : True, 'method_name': 'identity'})
    #  import sys ; sys.exit()
    # for testing
    from matplotlib import pyplot as plt

    def testing():
        dset = D.QualDR(img_transform=None, getitem_transform=None)

        #  dset2 = ietk.data.IDRiD('./data/IDRiD_segmentation')
        f, (ax1, ax2) = plt.subplots(1, 2)
        for id in torch.utils.data.RandomSampler(dset):
            dct = dset[id]
            im = np.array(dct['image'])
            #  fp = dct['fp']
            print(im.shape)

            im2 = preprocess(im, 'identity')
            im3 = preprocess(im, 'A+B+C+W+X')
            ax1.imshow(im2)
            ax2.imshow(im3)
            plt.pause(.1)
    testing()


        #  print(fp)
        #  f, (a,b, c) = plt.subplots(1, 3, num=1)
        #  #  a.imshow(fg, 'gray')
        #  b.imshow((im2.permute(1,2,0).numpy()*255).clip(0, 255).astype('uint8'))
        #  c.imshow(im)
        #  #  plt.show(block=True)
        #  #  plt.show(block=False)
        #  plt.pause(.01)
