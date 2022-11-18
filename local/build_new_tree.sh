#!/bin/bash

context_opts="--context-width=2 --central-position=1"
frame_subsampling_factor=3
other_opts=
num_leaves=6000
traindata=kaldi-s5/data/train
trainali="kaldi-s5/exp/i/tri4j_ali_train/"
validali="kaldi-s5/exp/i/tri4j_ali_parl-dev-all-fixed/"
langdir="kaldi-s5/data/lang_chain"
stage=0

. cmd.sh
cmd="$basic_cmd"

. path.sh
. parse_options.sh

if [ "$#" != 1 ]; then
  echo "Usage: $0 <treedir>"
  exit 1
fi

set -eu

treedir=$1

if [ $stage -le 0 ]; then
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor $frame_subsampling_factor \
    --context_opts "$context_opts" \
    $other_opts \
    $num_leaves \
    ${traindata} \
    ${langdir} \
    ${trainali} \
    ${treedir}
fi

if [ $stage -le 1 ]; then
  echo "Converting validation alignments"
  nj=$(cat ${validali}/num_jobs)
  $basic_cmd JOB=1:$nj ${treedir}/log/convert.valid.JOB \
    convert-ali --frame-subsampling-factor=$frame_subsampling_factor \
      ${validali}/final.mdl ${treedir}/1.mdl ${treedir}/tree \
      "ark:gunzip -c $validali/ali.JOB.gz|" "ark:|gzip -c >$treedir/ali.valid.JOB.gz" || exit 1;
fi
