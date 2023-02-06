#!/bin/bash

cmd="srun --mem 24G --time 2-0:0:0 -c5 --gres=gpu:1 --constraint volta -p dgx-spa,gpu,gpu-nvlink"
hparams="hyperparams/mtl/New-CRDNN-J-contd.yaml"
py_script="train-lfmmi-xent-fbank-mtl-outnorm-memfst.py"

. path.sh
. kaldi-s5/utils/parse_options.sh

timesfailed=0
while ! $cmd python $py_script $hparams; do
  timesfailed=$((timesfailed+1))
  if [ $timesfailed -le 5 ]; then
    echo "Training crashed, restarting!"
    sleep 3
  else
    echo "Crashed too many times, breaking!"
    break
  fi
done

