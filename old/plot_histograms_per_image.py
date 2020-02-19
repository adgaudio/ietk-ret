import numpy as np
from matplotlib import pyplot as plt
import os
from os.path import join
import argparse as ap
import multiprocessing as mp

import glob
import re
import competing_methods


def bap():
    p = ap.ArgumentParser()
    #  p.add_argument('--img-ids', nargs='*', help='By default, analyze all imgs.  If this parameter is given, only analyze given imgs:  --img-ids IDRiD_01 IMDiD_02')
    #  p.add_argument('--labels', nargs='*', default=('MA', 'HE', 'EX', 'SE', 'OD'), help='Lesions to analyze', choices=('MA', 'HE', 'EX', 'SE', 'OD'))
    #  p.add_argument('--methods', nargs='*', default=tuple(competing_methods.all_methods.keys()), choices=tuple(competing_methods.all_methods.keys()), help='list of methods named in competing_methods.all_methods')
    p.add_argument('--data-dir', default='./data/histograms_idrid_data', help='Location of the IDRiD dataset')
    p.add_argument('--save-dir', default='./data/histograms_idrid_plots/per_image', help='Location of the IDRiD dataset')
    #  p.add_argument('--show-plot', action='store_true', help='by default, dont show plot.  just save a figure to disk')
    return p


PATTERN = re.compile('(?P<img_id>IDRiD_\d{2})-(?P<lesion_name>..)-(?P<method_name>.*?).npz$')
color = ['red', 'green', 'blue']
lesion = ['EX', 'MA', 'SE', 'HE']


def save_img(fp):
    dat = np.load(fp)
    meta = PATTERN.search(fp).groupdict()
    key = f'{meta["img_id"]}-{meta["lesion_name"]}-{meta["method_name"]}'

    #  fig = plt.figure(num=lesion_id, figsize=(20,20))
    fig = plt.figure(num=1, figsize=(20,20))
    plt.clf()
    plt.cla()
    for ch in [0,1,2]:
        h = dat['healthy'][ch, :]
        d = dat['diseased'][ch, :]
        plt.plot(np.arange(256), h/h.sum(), '.', c=color[ch], alpha=.7, label=f'healthy -- {color[ch]}')
        plt.plot(np.arange(256), d/d.sum(), '.', c=color[ch], alpha=.3, label=f'diseased  -- {color[ch]}')
        plt.title(f'Histogram of Pixel intensities,\n{key}')
        plt.legend(loc='upper center')
        # #   fig.savefig(join(ns.save_dir, f'{name}-{lesion[lesion_id]}.png'))
        fig.savefig(join(ns.save_dir, f'{key}.png'))


if __name__ == "__main__":
    #  STYLE FOR A POSTER presentation
    #  SMALL_SIZE = 45
    #  MEDIUM_SIZE = 55
    #  BIGGER_SIZE = 65
    #  plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    #  plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    #  plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    #  plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #  plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #  plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    #  plt.rc('figure', titlesize=BIGGER_SIZE, figsize=(20,20))  # fontsize of the figure title
    #  plt.rc('lines', linewidth=10, markersize=25)

    ns = bap().parse_args()

    fps = glob.glob(join(f'{ns.data_dir}', '*.npz'))
    #  fps = ['data/histograms_idrid_data/IDRiD_39-MA-Unmodified_Image.npz',
        #  'data/histograms_idrid_data/IDRiD_39-MA-Illuminate_Sharpen.npz']


    os.makedirs(ns.save_dir, exist_ok=True)

    #  dats1 = [
        #  np.load('data/histograms_idrid_data/IDRiD_39-EX-Unmodified_Image.npz'),
        #  np.load('data/histograms_idrid_data/IDRiD_39-MA-Unmodified_Image.npz'),
        #  np.load('data/histograms_idrid_data/IDRiD_39-SE-Unmodified_Image.npz'),
        #  np.load('data/histograms_idrid_data/IDRiD_39-HE-Unmodified_Image.npz'),
    #  ]
    #  dats2 = [
        #  np.load('data/histograms_idrid_data/IDRiD_39-EX-Illuminate_Sharpen.npz'),
        #  np.load('data/histograms_idrid_data/IDRiD_39-MA-Illuminate_Sharpen.npz'),
        #  np.load('data/histograms_idrid_data/IDRiD_39-SE-Illuminate_Sharpen.npz'),
        #  np.load('data/histograms_idrid_data/IDRiD_39-HE-Illuminate_Sharpen.npz'),
    #  ]
    #  for name, datsX in [('IDRiD_39-Unmodified_Image', dats1), ('IDRiD_39-Illuminate-Sharpen', dats2)]:
        #  for lesion_id, dat in enumerate(datsX):
    with mp.Pool() as p:
        p.map(save_img, fps)
