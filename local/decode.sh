#!/bin/bash
# Script to test 
set -eu

am_cmd="srun --gres=gpu:1 --constraint=volta --time 1:0:0 --mem 24G -p dgx-common,dgx-spa,gpu,gpushort"
decode_cmd="slurm.pl --mem 12G --time 4:0:0"
score_cmd="slurm.pl --mem 2G --time 0:30:0"
nj=24
hparams="hyperparams/lfmmi/B.yaml"
datadir="kaldi-s5/data/parl-dev-all-fixed_hires/"
decodedir="exp/lfmmi-am/CRDNN-B/2602-2328units/decode-parl-dev-all"
tree="kaldi-s5/exp/chain/tree/"
graphdir="exp/chain/graph/graph_test/"
py_script=test-lfmmi.py
posteriors_from=

# Decoding params:
acwt=1.0
post_decode_acwt=10.0
beam=15
lattice_beam=8

# Script stage
stage=0

. path.sh
. parse_options.sh

posteriors_prefix="$decodedir/logprobs"
mkdir -p $decodedir

if [ $stage -le 1 ]; then
  if [[ $py_script == "test-xent.py" || $py_script == "test-lfmmi.py" ]]; then
    test_in="--test_feats "$datadir"/feats.scp"
  else
    test_in="--test_wavscp "$datadir"/wav.scp"
  fi
  $am_cmd python $py_script $hparams \
    $test_in \
    --test_probs_out "$posteriors_prefix".from_sb
  # Make SCPs:
  copy-matrix ark:"$posteriors_prefix".from_sb ark,scp:"$posteriors_prefix".ark,"$posteriors_prefix".scp
  utils/split_scp.pl "$posteriors_prefix".scp $(for n in `seq $nj`; do echo "$posteriors_prefix"."$n".scp; done)
fi

# Lattice generation
if [ $stage -le 2 ]; then 
  if [ -d $posteriors_from ]; then
    ln -s -f "$PWD"/"$posteriors_from"/logprobs* "$decodedir"/
  fi
  $decode_cmd JOB=1:$nj "$decodedir"/log/decode.JOB.log \
    latgen-faster-mapped \
    --acoustic-scale=$acwt \
    --beam=$beam \
    --lattice_beam=$lattice_beam \
    --word-symbol-table="$graphdir/words.txt" \
    $tree/final.mdl \
    "$graphdir/HCLG.fst" \
    scp:"$posteriors_prefix".JOB.scp \
    "ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c > $decodedir/lat.JOB.gz"
fi

if [ $stage -le 3 ]; then
  steps/score_kaldi.sh \
    --cmd "$score_cmd" \
    --beam $lattice_beam \
    "$datadir" \
    "$graphdir" \
    "$decodedir"
fi
