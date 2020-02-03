#!/usr/bin/env -S ipython --no-banner -i --
"""
Quick performance plots

$ ./thisfile.py -h
"""
import argparse as ap
import datetime as dt
import sys
import re
import glob
import os
from os.path import join, exists, basename
#  from os import exists
import pandas as pd
from matplotlib import pyplot as plt
#  import mpld3
#  import mpld3.plugins


def get_run_ids(ns):
    dirs = glob.glob(join(ns.data_dir, 'results/*'))
    for dir in dirs:
        run_id = basename(dir)
        if not re.search(ns.runid_regex, basename(dir)):
            continue
        yield run_id


def load_df_from_fp(fp, ns):
    print('found data', fp)
    if fp.endswith('.csv'):
        df = pd.read_csv(fp).set_index(ns.index_col)
    elif fp.endswith('.h5'):
        df = pd.read_hdf(fp, ns.hdf_table)
    return df


def _mode1_get_frames(ns):
    for run_id in get_run_ids(ns):
        #  if not exists(join(dir, 'lock.finished')):
            #  continue
        dirp = f'{ns.data_dir}/results/{run_id}'
        gen = (
            ((run_id, fname), load_df_from_fp(join(dirp, fname), ns))
            for fname in os.listdir(dirp)
            if re.search(ns.data_fp_regex, fname))
        yield from (x for x in gen if not x[-1].empty)


def make_plots(ns, cdfs):
    plot_cols = [col for col in cdfs.columns if re.search(ns.col_regex, col)]
    if not ns.no_savefig:
        fig, ax = plt.subplots(1,1,num=1)
    for col in plot_cols:
        if ns.no_savefig:
            fig, ax = plt.subplots(1,1)
        else:
            ax.cla()
        df = cdfs[col].unstack(('run_id', 'filename'))
        if ns.rolling_mean:
            df = df.rolling(ns.rolling_mean).mean()
        _plot_lines = df.plot(ax=ax, title=col)
        #  _legend = mpld3.plugins.InteractiveLegendPlugin(
            #  *_plot_lines.get_legend_handles_labels())
        #  mpld3.plugins.connect(fig, _legend)

        yield (fig, ax, col)


def savefig_with_symlink(fig, fp, symlink_fp):
    fig.savefig(fp, bbox_inches='tight')
    if os.path.islink(symlink_fp):
        os.remove(symlink_fp)
    prefix = os.path.dirname(os.path.commonprefix([fp, symlink_fp]))
    os.symlink(fp[len(prefix)+1:], symlink_fp)
    print('save fig', symlink_fp)


def main(ns):
    print(ns)
    # mode 1: compare each column across all files
    if ns.mode == 1:
        mode_1_plots(ns)
    elif ns.mode == 2:
        mode_2_plots(ns)
    else:
        raise Exception(f'not implemented mode: {ns.mode}')


def mode_1_plots(ns):
    dfs = dict(_mode1_get_frames(ns))
    print(dfs.keys())
    cdfs = pd.concat(dfs, sort=False, names=['run_id', 'filename'])
    globals().update({'cdfs_mode1': cdfs})

    timestamp = dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')  # date plot was created.  nothing to do with timestamp column.
    os.makedirs(join(ns.mode1_savefig_dir, 'archive'), exist_ok=True)
    for fig, ax, col in make_plots(ns, cdfs):
        if ns.no_savefig: continue
        savefig_with_symlink(
            fig,
            f'{ns.mode1_savefig_dir}/archive/{col}_{timestamp}.png',
            f'{ns.mode1_savefig_dir}/{col}_latest.png')


def mode_2_plots(ns):
    # mode 2: compare train to val performance
    cdfs_mode2 = {}
    timestamp = dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')  # date plot was created.  nothing to do with timestamp column.
    for run_id in get_run_ids(ns):
        dirp = f'{ns.data_dir}/results/{run_id}/log'
        if not os.path.exists(dirp):
            print('skip', run_id, 'contains no log data')
            continue

        cdfs = pd.concat({
            (run_id, fname): load_df_from_fp(join(dirp, fname), ns)
            for fname in os.listdir(dirp)
            if re.search(f'{ns.data_fp_regex}', fname)},
            sort=False, names=['run_id', 'filename'])
        cdfs_mode2[run_id] = cdfs

        os.makedirs(f'{ns.mode2_savefig_dir}/archive'.format(run_id=run_id), exist_ok=True)
        for fig, ax, col in make_plots(ns, cdfs):
            if ns.no_savefig: continue
            # save to file
            savefig_with_symlink(
                fig,
                f'{ns.mode2_savefig_dir}/archive/{col}_{timestamp}.png'.format(run_id=run_id),
                f'{ns.mode2_savefig_dir}/{col}_latest.png'.format(run_id=run_id))
    globals().update({'cdfs_mode2': cdfs_mode2})


def bap():
    par = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    A = par.add_argument
    A('runid_regex', help='find the run_ids that match the regular expression.')
    A('--data-dir', default='./data', help=' ')
    A('--data-fp-regex', default='perf.*\.csv', help=' ')
    #  A('--', nargs='?', default='one', const='two')
    A('--rolling-mean', '--rm', type=int, help='smoothing')
    A('--hdf-table-name', help='required if searching .h5 files')
    A('-c', '--col-regex', default='^(?!epoch|batch_idx|timestamp)', help='plot only columns matching regex.  By default, plot all except epoch and batch_idx.')
    A('--index-col', default='epoch', help=' ')
    A('--mode', default=1, type=int, choices=[1,2],
      help="`--mode 1` compares each column across all run_ids. `--mode 2` visualize the history of each column of a single run id.")
    A('--mode1-savefig-dir', default='./data/plots', help=' ')
    A('--mode2-savefig-dir', default='./data/results/{run_id}/plots', help=' ')
    A('--no-savefig', '--ns', action='store_true', help="don't save plots to file")
    return par


if __name__ == "__main__":
    main(bap().parse_args())
    print("type 'plt.show()' in terminal to see result")
