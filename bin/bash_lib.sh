# a library of helper functions for scripting
# shell scripts should source this library.


function run_parselog_py() {
  overwrite_plots="$1"
  fp_in="$2"  # a file in a subdirectory like run_id/data.log
  out_dir="$(dirname "$fp_in")"  # where to save plots
  if [ "${overwrite_plots:-false}" = false -a -e "$out_dir" ] ; then
    echo skipping $fp_in
    exit
  fi
  out="$(mktemp -p ./data/tmp/)"
  exec 3>"$out"
  exec 4<$out
  rm "$out"
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[0;33m'
  CYAN='\033[0;34m'
  NC='\033[0m' # No Color
  cmd="python ./bin/parselog.py ""$out_dir"" ""$fp_in"
  echo -e "$CYAN $cmd $NC"
  $cmd >&3 2>&3
  if [ $? -ne 0 ] ; then
    echo -e "$RED failed_to_parse $NC $fp_in"
    echo -e "$YELLOW "
    cat <&4
  else
    echo -e "$GREEN successfully_parsed $NC $fp_in"
    grep Traceback $fp_in >/dev/null && echo -e "$YELLOW    WARN: but log contains a Traceback $NC"
    echo "    output_dir $out_dir"
  fi
}
export -f run_parselog_py


# Helper function to ensure only one instance of a job runs at a time.
# Optionally, on finish, can write a file to ensure the job won't run again.
# usage: use_lockfile myfile.locked [ myfile.finished ]
function use_lockfile() {
  lockfile_fp="$(realpath -m ${1}.running)"
  lockfile_runonce="$2"
  lockfile_success_fp="$(realpath -m ${1}.finished)"
  lockfile_failed_fp="$(realpath -m ${1}.failed)"
  # create lock file
  if [ -e "$lockfile_fp" ] ; then
    echo "job already running!"
    exit
  fi
  if [ "$lockfile_runonce" = "yes" -a -e "$lockfile_success_fp" ] ; then
    echo "job previously completed!"
    exit
  fi
  mkdir -p "$(dirname "$lockfile_fp")"
  runid=$RANDOM
  echo $runid > "$lockfile_fp"

  # check that there wasn't a race condition
  # (not guaranteed to work but should be pretty good)
  sleep $(bc -l <<< "scale=4 ; ${RANDOM}/32767/10")
  rc=0
  grep $runid "$lockfile_fp" || rc=1
  if [ "$rc" = "1" ] ; then
    echo caught race condition 
    exit 1
  fi

  # before starting current job, remove evidence of failed job.
  if [ -e "$lockfile_failed_fp" ] ; then
    rm "$lockfile_failed_fp"
  fi

  # automatically remove the lockfile when finished, whether fail or success
  function remove_lockfile() {
    rm $lockfile_fp
  }
  function trap_success() {
    if [ ! -e "$lockfile_failed_fp" ] ; then
      echo job successfully completed
      if [ "$lockfile_runonce" = "yes" ] ; then
        echo please rm this file to re-run job again: ${lockfile_success_fp}
        date > $lockfile_success_fp
        hostname >> $lockfile_success_fp
      fi
    fi
    remove_lockfile
    exit 0
  }
  function trap_err() {
    rv=$?
    echo "ERROR code=$rv" >&2
    date > $lockfile_failed_fp
    exit $rv
  }
  trap trap_success EXIT  # always run this
  trap trap_err ERR
  trap trap_err INT
}
export -f use_lockfile


function log_initial_msgs() {(
  set -eE
  set -u
  run_id=$1
  echo "Running on hostname: $(hostname)"
  echo "run_id: ${run_id}"
  date

  # print out current configuration
  echo ======================
  echo CURRENT GIT CONFIGURATION:
  echo "git commit: $(git rev-parse HEAD)"
  echo
  echo git status:
  git status
  echo
  echo git diff:
  git --no-pager diff --cached
  git --no-pager diff
  echo
  echo ======================
  echo
  echo
)}
export -f log_initial_msgs


function run_cmd() {
  run_id="$1"
  cmd="$2"
  export -f log_initial_msgs
cat <<EOF | bash
set -eE
set -u
log_initial_msgs $run_id
echo run_id="$run_id" "$cmd"
run_id="$run_id" $cmd
echo job finished
date
EOF
}
export -f run_cmd


function run_cmd_and_log() {
  run_id="$1"
  shift
  cmd="$@"
  lockfile_path=./data/results/$run_id/lock
  lockfile_runonce=yes

  (
  set -eE
  set -u
  set -o pipefail
  use_lockfile ${lockfile_path} ${lockfile_runonce}
  log_fp="./data/results/$run_id/`date +%Y%m%dT%H%M%S`.log"
  mkdir -p "$(dirname "$(realpath -m "$log_fp")")"
  run_cmd "$run_id" "$cmd" 2>&1 | tee $log_fp
  )
}
export -f run_cmd_and_log


function run() {
  local ri="$1"
  shift
  local cmd="$@"
  run_cmd_and_log $ri $@
}
export -f run


function fork() {
  (run $@) &
}
export -f fork



function round_robbin_gpu() {
  # you probably want to use `run_gpus` instead.
  local num_gpus=$(nvidia-smi pmon -c 1|grep -v \# | awk '{print $1}' | sort -u | wc -l)
  local num_tasks=${1:-$num_gpus}
  local idx=0

  while read -r line0 ; do

    local gpu_idx=$(( $idx % num_gpus ))
    device=cuda:$gpu_idx fork $line0
    local idx=$(( ($idx + 1) % $num_tasks ))
    if [ $idx = 0 ] ; then
      wait # ; sleep 5
    fi
  done
  if [ $idx != 0 ] ; then
    wait # ; sleep 5
  fi
}
export -f round_robbin_gpu


function run_gpus() {
  # use redis database as a queuing mechanism.  you can specify how to connect to redis with RUN_GPUS_REDIS_CLI 
  local redis="${RUN_GPUS_REDIS_CLI:-redis-cli -n 1}"
  local num_gpus=$(nvidia-smi pmon -c 1|grep -v \# | awk '{print $1}' | sort -u | wc -l)
  local Q="`mktemp -u -p run_gpus`"

  trap "$(echo $redis DEL "$Q" "$Q/numstarted") > /dev/null" EXIT

  # --> publish to the redis queue
  local maxjobs=0
  while read -r line0 ; do
    $redis LPUSH "$Q" "$line0" >/dev/null
    local maxjobs=$(( $maxjobs + 1 ))
  done
  $redis EXPIRE "$Q" 1209600 >/dev/null # expire queue after two weeks in case trap fails. should make all the rpush events and this expire atomic, but oh well.
  # --> start the consumers
  for gpu_idx in `nvidia-smi pmon -c 1|grep -v \# | awk '{print $1}' | sort -u` ; do
    consumergpu_redis $gpu_idx "$redis" "$Q" $maxjobs &
  done
  wait
  $redis DEL "$Q" "$Q/numstarted" >/dev/null
}


function consumergpu_redis() {
  local gpu_idx=$1
  local redis="$2"
  local Q="$3"
  local maxjobs=$4

  while /bin/true ; do
    # --> query redis for a job to run
    rv="$($redis --raw <<EOF
MULTI
INCR $Q/numstarted
EXPIRE "$Q" 1209600
EXPIRE "$Q/numstarted" 1209600
RPOPLPUSH $Q $Q
EXEC
EOF
)"
    local cmd="$( echo "$rv" | tail -n 1)"
    local num_started="$( echo "$rv" | tail -n 4| head -n 1)"
    # --> run the job if it hasn't been started previously
    if [ "$num_started" -le "$maxjobs" ] ; then
      device=cuda:$gpu_idx run "$cmd"
    else
      break
    fi
  done
}
export -f run_gpus
