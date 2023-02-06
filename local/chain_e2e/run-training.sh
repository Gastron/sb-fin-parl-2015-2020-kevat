#!/bin/bash

cmd="srun --mem 24G --time 2-0:0 -c5 --gres=gpu:1 --constraint volta -p dgx-spa,gpu,gpu-nvlink"
hparams="hyperparams/mtl/w2v2-C.yaml"
treedir="exp/chain/tree2/"
py_script="local/chain_e2e/sb-train-lfmmi-e2e.py"

. path.sh
. parse_options.sh

num_units=$(tree-info $treedir/tree | grep "num-pdfs" | cut -d" " -f2)

timesfailed=0
while ! $cmd python $py_script $hparams --num_units $num_units; do
  timesfailed=$((timesfailed+1))
  if [ $timesfailed -le 5 ]; then
    echo "Training crashed, restarting!"
    sleep 3
  else
    echo "Crashed too many times, breaking!"
    break
  fi
done

