#!/usr/bin/env bash
# a script (run on the local computer) to reproduce experiments.

set -e
set -u

cd "$(dirname "$(dirname "$(realpath "$0")")")"
pwd

. ./bin/bash_lib.sh


# how to run jobs
# - this can run locally on your computer.
# - `run_gpus` takes care of logging, race conditions and lock files, and making use of multiple gpus.
#    In particular, it sets an env var (device=cuda:X) asking the screendr project to use a particular gpu.
# - test_my_run_id is the name of your experiment
# your experiments here, identified by name.  The results show up in ./data/results/test_my_run_id and ./data/results/test_my_other_experiment

# find a learning rate
# (
# for lr in 0.0001 0.0005 0.001 0.005 0.01 ; do
# for lr in 0.0001 0.0005 0.001 ; do
# echo lr$lr python -m screendr model_configs BDSQualDR --ietk-method-name identity \
  # --optimizer-lr $lr \
  # --epochs 100 --debug-small-dataset 100
# done
# ) | run_gpus
# # --> conc: let's fix lr to 0.0002

# # find a weight decay
# (
# for wd in 0 0.001 0.0001 0.00001 0.1 ; do
  # echo wd$wd python -m screendr model_configs BDSQualDR --ietk-method-name identity \
    # --optimizer-weight-decay $wd \
    # --epochs 100 --debug-small-dataset 100
  # done
# ) | run_gpus
# --> conc: use 0.0001

# device="cuda:0" run identity python -m screendr model_configs BDSQualDR --ietk-method-name identity

## Does clipping image into 0,1 range improve perf?  Might if model is pre-trained on clipped imgs...
# (
# echo test-mul255 python -m screendr model_configs BDSQualDR --ietk-method-name A --epochs 100 --clip-imgs --mul255test
# for mdl in A A2 B C C2 C3 D W X Y Z A+X C+X A+C A+Z A+C+X A+C+X+Z identity; do
# for mdl in A+B B+C B+X A+B+C B+C+X A+B+C+X ; do
#   echo Qclip-$mdl python -m screendr model_configs BDSQualDR --ietk-method-name $mdl --preprocess-clip-imgs --no-preprocess-mul255 --epochs 50
#   # echo Qclipmul-$mdl python -m screendr model_configs BDSQualDR --ietk-method-name $mdl --preprocess-clip-imgs --preprocess-mul255 --epochs 50
#   # echo Qnoclipmul-$mdl python -m screendr model_configs BDSQualDR --ietk-method-name $mdl --no-preprocess-clip-imgs --preprocess-mul255 --epochs 50
#   # echo Qnoclipnomul-$mdl python -m screendr model_configs BDSQualDR --ietk-method-name $mdl --no-preprocess-clip-imgs --no-preprocess-mul255 --epochs 50
#   done
# ) | run_gpus
# conc: 120 epochs is too many - only need 50.
# conc: is mul255 better or not.  don't mul 255 by far is best.
# conc: clip vs no clip: clip by far is best.

models="A A2 B C C2 C3 D W X Y Z A+X C+X A+C A+Z A+C+X A+C+X+Z A+B sA+sB2 B+C B+X A+B+C A+B+X B+C+X A+B+C+X A+B+C+W+X sA+sX sC+sX sA+sC sA+sZ sA+sC+sX sA+sC+sX+sZ sA+sB sB+sC sB+sX sA+sB+sC sA+sB+sX sB+sC+sX sA+sB+sC+sX sA+sB+sC+sW+sX identity"

# re-train with smarter early stopping and checkpoint logging.
# and also evaluate on test set.
# (
# # for mdl in A2 sA+sB A B C D W X Y Z A+X C+X A+C A+Z A+C+X A+C+X+Z A+B B+C B+X A+B+C A+B+X B+C+X A+B+C+X identity ; do
# for mdl in $models ; do
#   echo Q1-$mdl python -m screendr model_configs BDSQualDR --ietk-method-name $mdl --epochs 120 --checkpoint-fname 'epoch_best.pth'
#   # echo Qtest1-$mdl python -m screendr model_configs BDSQualDR \
#   #   --ietk-method-name $mdl \
#   #   --checkpoint-fp ./data/results/Q1-$mdl/model_checkpoints/epoch_best.pth \
#   #   --no-data-use-train-set
# done
# for mdl in $models ; do
#   # echo Qtest1.2-$mdl python -m screendr model_configs BDSQualDR \
#   #   --ietk-method-name $mdl \
#   #   --checkpoint-fp ./data/results/Q1-$mdl/model_checkpoints/epoch_best.pth \
#   #   --no-data-use-train-set
#   echo Qtest1.3-$mdl python -m screendr model_configs BDSQualDR \
#     --ietk-method-name $mdl \
#     --checkpoint-fp ./data/results/Q1-$mdl/model_checkpoints/epoch_best.pth \
#     --no-data-use-train-set
# done
# ) | run_gpus


# now run the model with class balancing weights.
# fix a bug where validation set was evaluated with cutout.
# run all models with the same early stopping.
# good enough for final results now.
(
for mdl in $models ; do
  echo Q2-$mdl python -m screendr model_configs BDSQualDR \
    --ietk-method-name $mdl --epochs 120 --checkpoint-fname 'epoch_best.pth'
done
for mdl in $models ; do
  echo Qtest2-$mdl python -m screendr model_configs BDSQualDR \
    --ietk-method-name $mdl \
    --checkpoint-fp ./data/results/Q2-$mdl/model_checkpoints/epoch_best.pth \
    --no-data-use-train-set
done
) | run_gpus
# TODO: do I also want some results with W models?
