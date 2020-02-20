#!/usr/bin/env bash
# a script (run on the local computer) to reproduce experiments.

set -e
set -u

# ensure running from repo root.
cd "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"
pwd

. ./bin/bash_lib.sh


# how to run jobs
# - this can run locally on your computer.
# - `run_gpus` takes care of logging, race conditions and lock files, and making use of multiple gpus.
#    In particular, it sets an env var (device=cuda:X) asking the simplepytorch project to use a particular gpu.
# - test_my_run_id is the name of your experiment
# your experiments here, identified by name.  The results show up in ./data/results/test_my_run_id and ./data/results/test_my_other_experiment


models="identity A A2 B C C2 C3 D W X Y Z A+X C+X A+C A+Z A+C+X A+C+X+Z A+B B+C B+X A+B+C A+B+X B+C+X A+B+C+X A+B+C+W+X sA+sX sC+sX sA+sC sA+sZ sA+sC+sX sA+sC+sX+sZ sA+sB sB+sC sB+sX sA+sB+sC sA+sB+sX sB+sC+sX sA+sB+sC+sX sA+sB+sC+sW+sX"
(
# for mdl in $models ; do
#   echo R1-$mdl python -m simplepytorch model_configs BDSSegment \
#     --data-name rite --ietk-method-name $mdl \
#     --epochs 120 \
#     --checkpoint-fname 'epoch_best.pth'
# done

# fix bug in sharpen that it doesn't work for small imgs.
# fix bugs in preprocessing steps
# for mdl in $models ; do
#   echo I1-$mdl python -m simplepytorch model_configs BDSSegment \
#     --data-name idrid --ietk-method-name $mdl \
#     --epochs 100 \
#     --checkpoint-fname 'epoch_best.pth'
# done
# for mdl in $models ; do
#   echo R1.3-$mdl python -m simplepytorch model_configs BDSSegment \
#     --data-name rite --ietk-method-name $mdl \
#     --epochs 80 \
#     --checkpoint-fname 'epoch_best.pth'
# done

# okay now train on train set and evaluate test set.
# bigger model, more metrics, rite also does av predictions, save imgs with checkpoints
# R2 and I2 have no class balancing weights and a MCC bug.  the I2.2 and R2.2 address them

for mdl in $models ; do
  # echo I2.2-$mdl python -m simplepytorch model_configs BDSSegment \
    # --data-name idrid --ietk-method-name $mdl \
    # --epochs 100 \
    # --checkpoint-fname 'epoch_best.pth'
  echo R2.2-$mdl python -m simplepytorch model_configs BDSSegment \
    --data-name rite --ietk-method-name $mdl \
    --epochs 100 \
    --checkpoint-fname 'epoch_best.pth'
done

# for mdl in $models ; do
  # echo Itest2-$mdl python -m simplepytorch model_configs BDSSegment \
    # --data-name idrid --ietk-method-name $mdl \
    # --epochs 100 \
    # --checkpoint-fp ./data/results/I2.2-$mdl/model_checkpoints/epoch_best.pth \
    # --no-data-use-train-set
# done
for mdl in $models ; do
  echo Rtest2-$mdl python -m simplepytorch model_configs BDSSegment \
    --data-name rite --ietk-method-name $mdl \
    --epochs 100 \
    --checkpoint-fp ./data/results/R2.2-$mdl/model_checkpoints/epoch_best.pth \
    --no-data-use-train-set
done
) | run_gpus

# when testing, set this to 1:
    # --data-train-val-split 0.8
