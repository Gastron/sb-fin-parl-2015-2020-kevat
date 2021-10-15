#!/bin/bash
# Script to test Xent trained model
set -eu

am_cmd="srun --gres=gpu:1 --constraint=volta --time 1:0:0 --mem 24G -p dgx-common,dgx-spa,gpu,gpushort"
decode_cmd="slurm.pl --mem 2G"
nj=8
hparams="hyperparams/xent/A.yaml"
datadir="kaldi-s5/data/parl-dev-all-fixed_hires/"
decodedir="exp/xent-am/CRDNN-A/2602-2328units/decode-parl-dev-all"
lang="kaldi-s5/data/lang_test_varikn.bpe1750.d0.0001/"
tree="kaldi-s5/exp/chain/tree/"
graphdir="exp/chain/graph/graph_xent"

# Decoding params:
acwt=0.1
beam=15
lattice_beam=8

# Script stage
stage=0

. path.sh
. parse_options.sh

posteriors_prefix="$decodedir/logprobs"
mkdir -p $decodedir

if [ $stage -le 0 ]; then
  # Graph
  utils/mkgraph.sh $lang $tree $graphdir
fi

if [ $stage -le 1 ]; then
  $am_cmd python test-hmm-dnn.py $hparams \
    --test_feats "$datadir"/feats.scp \
    --test_probs_out "$posteriors_prefix".from_sb
  # Make SCPs:
  copy-matrix ark:"$posteriors_prefix".from_sb ark,scp:"$posteriors_prefix".ark,"$posteriors_prefix".scp
  utils/split_scp.pl "$posteriors_prefix".scp $(for n in `seq $nj`; do echo "$posteriors_prefix"."$n".scp; done)
fi

# Lattice generation
if [ $stage -le 2 ]; then 
  $decode_cmd JOB=1:$nj "$decodedir"/log/decode.JOB.log \
    latgen-faster-mapped \
    --acoustic-scale=$acwt \
    --beam=$beam \
    --lattice_beam=$lattice_beam \
    --word-symbol-table="$graphdir/words.txt" \
    $tree/final.mdl \
    "$graphdir/HCLG.fst" \
    scp:"$posteriors_prefix".JOB.scp \
    "ark:| gzip -c > $decodedir/lat.JOB.gz"
fi

if [ $stage -le 3 ]; then
  steps/score_kaldi.sh \
    --cmd "$decode_cmd" \
    --beam $lattice_beam \
    "$datadir" \
    "$graphdir" \
    "$decodedir"
fi
