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
#    In particular, it sets an env var (device=cuda:X) asking the screendr project to use a particular gpu.
# - test_my_run_id is the name of your experiment
# your experiments here, identified by name.  The results show up in ./data/results/test_my_run_id and ./data/results/test_my_other_experiment


# using a subset of models from 
models="A A2 B C C2 C3 D W X Y Z A+X C+X A+C A+Z A+C+X A+C+X+Z A+B sA+sB2 B+C B+X A+B+C A+B+X B+C+X A+B+C+X A+B+C+W+X sA+sX sC+sX sA+sC sA+sZ sA+sC+sX sA+sC+sX+sZ sA+sB sB+sC sB+sX sA+sB+sC sA+sB+sX sB+sC+sX sA+sB+sC+sX sA+sB+sC+sW+sX identity"
(
for mdl in $models ; do
  echo R1-$mdl python -m screendr model_configs BDSSegment \
    --data-name rite --ietk-method-name $mdl \
    --epochs 120 \
    --checkpoint-fname 'epoch_best.pth'
done
# for mdl in $models ; do
  # echo Qtest2-$mdl python -m screendr model_configs BDSQualDR \
    # --ietk-method-name $mdl \
    # --no-data-use-train-set
# done
) | run_gpus
# TODO: do I also want some results with W models?

# when testing, set this to 1:
    # --data-train-val-split 0.8
