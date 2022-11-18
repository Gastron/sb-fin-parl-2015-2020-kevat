#!/bin/bash
# NOTE:
# These are the commands I ran in kaldi-s5 to produce the validation set alignments.
# This file is just meant as a reference, not to be run directly

source path.sh

steps/align_fmllr.sh --nj 16 --cmd "slurm.pl --mem 4G --time 2:0:0" data/parl-dev-all-fixed data/lang_train_and_valid exp/i/tri4j exp/i/tri4j_ali_parl-dev-all-fixed

slurm.pl --mem 2G --time 0:30:0 JOB=1:16 exp/chain/tree/log/convert.valid.JOB.log \
  convert-ali --frame-subsampling-factor=3 \
  exp/i/tri4j_ali_parl-dev-all-fixed/final.mdl exp/chain/tree/1.mdl exp/chain/tree/tree \
  "ark:gunzip -c exp/i/tri4j_ali_parl-dev-all-fixed/ali.JOB.gz|" "ark:|gzip -c >exp/chain/tree/ali.parl-dev-all-fixed.JOB.gz"
