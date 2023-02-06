#!/bin/bash
set -eu

stage=3

. path.sh
. cmd.sh
. utils/parse_options.sh

if [ $stage -le 3 ]; then
  local/chain_e2e/build_new_tree.sh \
    --type biphone \
    --min_biphone_count 100 \
    --min_monophone_count 10 \
    --tie true \
    kaldi-s5/data/parl2015-2020-train/ \
    kaldi-s5/data/lang_chain \
    exp/chain_e2e/tree
fi

num_units=$(tree-info exp/chain_e2e/tree/tree | grep "num-pdfs" | cut -d" " -f2)
seed=2602

if [ $stage -le 5 ]; then
  local/prepare_graph_clustered.sh \
    --dataroot kaldi-s5/data/ \
    --train_set parl2015-2020-train \
    --valid_set parl-dev-all-fixed \
    --lang kaldi-s5/data/lang_chain \
    --tree_dir exp/chain_e2e/tree \
    --graph exp/chain_e2e/graph
fi

if [ $stage -le 6 ]; then
  local/chain_e2e/run-training.sh \
    --treedir exp/chain_e2e/tree \
    --hparams "hyperparams/lfmmi/New-CRDNN-J-e2e.yaml"
fi

if [ $stage -le -999 ]; then
  # Document how the non Flat Start LF-MMI-only model was trained:
  local/chain_e2e/run-training.sh \
    --treedir kaldi-s5/exp/chain/tree/ \
    --hparams "hyperparams/lfmmi/New-CRDNN-J-e2e-nonflat.yaml"
  local/chain-decode.sh \
    --hparams hyperparams/lfmmi/New-CRDNN-J-e2e-nonflat.yaml \
    --decodedir exp/lfmmi-am/New-CRDNN-J-nonflat/${seed}-${num_units}units/decode_parl-dev-all-fixed_varikn.bpe1750.d0.0001_acwt1.3 \
    --py_script "test-lfmmi-fbank.py"
fi


if [ $stage -le 7 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 kaldi-s5/data/lang_test_varikn.bpe1750.d0.0001/ exp/chain_e2e/graph exp/chain_e2e/graph/graph_varikn.bpe1750.d0.0001
fi

if [ $stage -le 8 ]; then
  local/chain-decode.sh \
    --tree exp/chain_e2e/tree \
    --graphdir exp/chain_e2e/graph/graph_varikn.bpe1750.d0.0001/ \
    --hparams hyperparams/lfmmi/New-CRDNN-J-e2e.yaml \
    --decodedir exp/lfmmi-am/New-CRDNN-J/${seed}-${num_units}units/decode_parl-dev-all-fixed_varikn.bpe1750.d0.0001_acwt1.3 \
    --py_script "test-lfmmi-fbank.py"
fi

if [ $stage -le 9 ]; then
  local/chain_e2e/run-training.sh \
    --treedir exp/chain_e2e/tree \
    --hparams "hyperparams/lfmmi/New-CRDNN-J-e2e-contd.yaml"
fi

if [ $stage -le 10 ]; then
  local/chain-decode.sh \
    --tree exp/chain_e2e/tree \
    --graphdir exp/chain_e2e/graph/graph_varikn.bpe1750.d0.0001/ \
    --hparams hyperparams/lfmmi/New-CRDNN-J-e2e-contd.yaml \
    --decodedir exp/lfmmi-am/New-CRDNN-J-contd/${seed}-${num_units}units/decode_parl-dev-all-fixed_varikn.bpe1750.d0.0001_acwt1.3 \
    --py_script "test-lfmmi-fbank.py"
fi

