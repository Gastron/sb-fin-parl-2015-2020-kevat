#!/bin/bash
# Copyright (c) Yiwen Shao, Aku Rouhe
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e -o pipefail

stage=0
train_set=train_clean_5
valid_set=dev_clean_2
rootdir=data

nj=10

tree_dir=exp/chain/tree
graph=exp/chain/graph
langdir=data/lang
lang=data/lang/lang_chain

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh


if [ $stage -le 0 ]; then
  echo "$0: Stage 0: Phone LM estimating"
  echo "Estimating a phone language model for the denominator graph..."
  mkdir -p $graph/log
  $train_cmd $graph/log/make_phone_lm.log \
             cat $rootdir/$train_set/text \| \
             steps/nnet3/chain/e2e/text_to_phones.py --between-silprob 0.1 \
             $lang \| \
             utils/sym2int.pl -f 2- $lang/phones.txt \| \
             chain-est-phone-lm --num-extra-lm-states=2000 \
             ark:- $graph/phone_lm.fst
fi

if [ $stage -le 1 ]; then
  echo "$0: Stage 1: Graph generation..."
  echo "Copying the relevant files"
  cp $tree_dir/final.mdl $graph/0.mdl
  cp $tree_dir/final.mdl $graph/final.mdl
  copy-transition-model $tree_dir/final.mdl $graph/0.trans_mdl
  cp $tree_dir/tree $graph/tree
  echo "Making denominator graph..."
  $train_cmd $graph/log/make_den_fst.log \
	     chain-make-den-fst $graph/tree $graph/final.mdl \
	     $graph/phone_lm.fst \
	     $graph/den.fst $graph/normalization.fst
fi


if [ $stage -le 2 ]; then
  echo "Making numerator graph..."
  lex=$lang/L.fst
  oov_sym=`cat $lang/oov.int` || exit 1;
  for x in $train_set $valid_set; do
    sdata=$rootdir/$x/split$nj;
    [[ -d $sdata && $rootdir/$x/feats.scp -ot $sdata ]] || split_data.sh $rootdir/$x $nj || exit 1;
    $train_cmd JOB=1:$nj $graph/$x/log/compile_graphs.JOB.log \
    	       compile-train-graphs $scale_opts --read-disambig-syms=$lang/phones/disambig.int \
    	       $graph/tree $graph/final.mdl $lex \
    	       "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $sdata/JOB/text|" \
    	       "ark,scp:$graph/$x/fst.JOB.ark,$graph/$x/fst.JOB.scp" || exit 1;
    $train_cmd JOB=1:$nj $graph/$x/log/make_num_fst.JOB.log \
    	       chain-make-num-fst-e2e $graph/final.mdl $graph/normalization.fst \
    	       scp:$graph/$x/fst.JOB.scp ark,scp:$graph/$x/num.JOB.ark,$graph/$x/num.JOB.scp
    for id in $(seq $nj); do cat $graph/$x/num.$id.scp; done > $graph/$x/num.scp
  done
fi

if [ $stage -le 3 ]; then
  echo "Making HCLG full graph..."
  utils/lang/check_phones_compatible.sh \
    $langdir/lang_nosp_test_tgsmall/phones.txt $lang/phones.txt
  utils/mkgraph.sh \
    --self-loop-scale 1.0 $langdir/lang_nosp_test_tgsmall \
    $graph $graph/graph_tgsmall || exit 1;
fi
