#!/bin/bash

min_lmwt=5
max_lmwt=20
lm_name="trafo_transcript"
wer_hyp_filter="local/wer_hyp_filter"

. path.sh
. parse_options.sh


if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <nbest-dir>"
  exit 1 
fi

nbestdir=$1
reffile=$nbestdir/../scoring_kaldi/test_filt.txt

if [ ! -f $reffile ]; then
  echo "Expected $reffile to exist!"
  exit 1
fi

mkdir -p $nbestdir/scoring

for lmwt in $(seq $min_lmwt $max_lmwt); do
  local/rescore_nbest.py --lm-weight $lmwt \
    $nbestdir/text \
    $nbestdir/ac_cost \
    $nbestdir/lm_cost.$lm_name \
    > $nbestdir/scoring/rescored_text.lmwt${lmwt}.$lm_name
  compute-wer \
    ark:"$reffile" \
    ark:"$wer_hyp_filter <$nbestdir/scoring/rescored_text.lmwt${lmwt}.$lm_name |" \
    > $nbestdir/scoring/wer.lmwt${lmwt}.$lm_name 2>/dev/null
done

for lmwt in $(seq $min_lmwt $max_lmwt); do
  # adding /dev/null to the command list below forces grep to output the filename
  grep WER $nbestdir/scoring/wer.lmwt${lmwt}.$lm_name /dev/null
done | utils/best_wer.sh >& $nbestdir/scoring/best_wer.$lm_name

