#!/bin/bash

cmd="srun --mem 8G --time 4:0:0 -c 2"
datadir="data/uit-sme-segmented-and-giellagas-train/"
sharddir="shards/train-sub2"
nj=8

. path.sh
. parse_options.sh

$cmd local/chain/make_shards.py $nj "$sharddir" \
  --num-proc 0 \
  --segments "$datadir"/split$nj/JOB/segments \
             "$datadir"/split$nj/JOB/wav.scp 

