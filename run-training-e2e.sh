#!/bin/bash
stage=1


. path.sh
. cmd.sh
. utils/parse_options.sh


if [ $stage -le 10 ]; then
  local/chain_e2e/run-training.sh \
    --treedir exp/chain_e2e/tree \
    --hparams "hyperparams/lfmmi/FIX-New-CRDNN-J-e2e.yaml"
fi

if [ $stage -le 11 ]; then
  local/chain_e2e/run-training.sh \
    --treedir exp/chain_e2e/tree \
    --hparams "hyperparams/lfmmi/FIX-New-CRDNN-J-e2e-contd.yaml"

fi

if [ $stage -le 12 ]; then
  local/chain-decode.sh \
    --graphdir "exp/chain_e2e//graph/sb-vocab-train20-varikn.d0.0001-bpe1750/" \
    --tree exp/chain_e2e/tree \
    --hparams "hyperparams/lfmmi/FIX-New-CRDNN-J-e2e.yaml" \
    --py_script "test-lfmmi-fbank.py" \
    --decodedir "exp/lfmmi-am/FIX-New-CRDNN-J/2602-1128units/decode_parl-dev-all-fixed"
fi

if [ $stage -le 13 ]; then
  local/chain-decode.sh \
    --tree exp/chain_e2e/tree \
    --graphdir "exp/chain_e2e//graph/sb-vocab-train20-varikn.d0.0001-bpe1750/" \
    --hparams "hyperparams/lfmmi/FIX-New-CRDNN-J-e2e-contd.yaml" \
    --py_script "test-lfmmi-fbank.py" \
    --decodedir "exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-dev-all-fixed"
fi

if [ $stage -le 14 ]; then
  local/chain-decode.sh \
    --datadir data/parl-test-all \
    --tree exp/chain_e2e/tree \
    --graphdir "exp/chain_e2e//graph/sb-vocab-train20-varikn.d0.0001-bpe1750/" \
    --hparams "hyperparams/lfmmi/FIX-New-CRDNN-J-e2e-contd.yaml" \
    --py_script "test-lfmmi-fbank.py" \
    --decodedir "exp/lfmmi-am/FIX-New-CRDNN-J-contd//2602-1128units//decode_parl-test-all"
fi

if [ $stage -le 15 ]; then
  local/chain-decode.sh \
    --datadir kaldi-s5/data/parl2020-test/ \
    --tree exp/chain_e2e/tree \
    --graphdir "exp/chain_e2e//graph/sb-vocab-train20-varikn.d0.0001-bpe1750/" \
    --hparams "hyperparams/lfmmi/FIX-New-CRDNN-J-e2e-contd.yaml" \
    --py_script "test-lfmmi-fbank.py" \
    --decodedir "exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units/decode_parl-test-2020"
fi





if [ $stage -le 16 ]; then
  local/chain-decode.sh \
    --tree exp/chain_e2e/tree \
    --graphdir "exp/chain_e2e//graph/sb-vocab-parl30M-varikn.d0.0001-bpe1750/" \
    --hparams "hyperparams/lfmmi/FIX-New-CRDNN-J-e2e-contd.yaml" \
    --py_script "test-lfmmi-fbank.py" \
    --stage 2 \
    --decodedir "exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750/"
fi

if [ $stage -le 17 ]; then
  local/chain-decode.sh \
    --datadir data/parl-test-all \
    --tree exp/chain_e2e/tree \
    --graphdir "exp/chain_e2e//graph/sb-vocab-parl30M-varikn.d0.0001-bpe1750/" \
    --hparams "hyperparams/lfmmi/FIX-New-CRDNN-J-e2e-contd.yaml" \
    --py_script "test-lfmmi-fbank.py" \
    --decodedir "exp/lfmmi-am/FIX-New-CRDNN-J-contd//2602-1128units//decode_parl-test-all_sb-vocab-parl30M-varikn.d0.0001-bpe1750/"
fi

if [ $stage -le 18 ]; then
  local/chain-decode.sh \
    --datadir kaldi-s5/data/parl2020-test/ \
    --tree exp/chain_e2e/tree \
    --graphdir "exp/chain_e2e//graph/sb-vocab-parl30M-varikn.d0.0001-bpe1750/" \
    --hparams "hyperparams/lfmmi/FIX-New-CRDNN-J-e2e-contd.yaml" \
    --py_script "test-lfmmi-fbank.py" \
    --decodedir "exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units/decode_parl-test-2020_sb-vocab-parl30M-varikn.d0.0001-bpe1750/"
fi



LMWT=$(cat exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750/scoring_kaldi/wer_details/lmwt)
if [ $stage -le 19 ]; then
  #echo 8 > exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750/num_jobs
  local/get_nbest.sh \
    --num_best 100 \
    --LMWT $LMWT \
    exp/chain_e2e/graph/sb-vocab-parl30M-varikn.d0.0001-bpe1750 \
    exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750/ \
    exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units/decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best

  #echo 8 > exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-all_sb-vocab-parl30M-varikn.d0.0001-bpe1750/num_jobs
  local/get_nbest.sh \
    --num_best 100 \
    --LMWT $LMWT \
    exp/chain_e2e/graph/sb-vocab-parl30M-varikn.d0.0001-bpe1750 \
    exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-all_sb-vocab-parl30M-varikn.d0.0001-bpe1750 \
    exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-all_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best

  #echo 8 > exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-2020_sb-vocab-parl30M-varikn.d0.0001-bpe1750/num_jobs
  local/get_nbest.sh \
    --num_best 100 \
    --LMWT $LMWT \
    exp/chain_e2e/graph/sb-vocab-parl30M-varikn.d0.0001-bpe1750 \
    exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-2020_sb-vocab-parl30M-varikn.d0.0001-bpe1750 \
    exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-2020_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best
fi

if [ $stage -le 20 ]; then
  pushd ../speechbrain_2015-2020-kevat_e2e
  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best

  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-all_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best

  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-2020_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best
  popd
fi

if [ $stage -le 21 ]; then
  local/run_rescoring.sh --lm_name trafo_30M exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best
  local/run_rescoring.sh --lm_name trafo_30M exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-all_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best
  local/run_rescoring.sh --lm_name trafo_30M exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-2020_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best
  exit
fi


LMWT=$(cat exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units/decode_parl-dev-all-fixed/scoring_kaldi/wer_details/lmwt)
if [ $stage -le 22 ]; then
  local/get_nbest.sh \
    --num_best 100 \
    --LMWT $LMWT \
    exp/chain_e2e/graph/sb-vocab-train20-varikn.d0.0001-bpe1750/ \
    exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-dev-all-fixed/ \
    exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-dev-all-fixed/100-best

  #echo 8 > exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-all/num_jobs
  local/get_nbest.sh \
    --num_best 100 \
    --LMWT $LMWT \
    exp/chain_e2e/graph/sb-vocab-train20-varikn.d0.0001-bpe1750/ \
    exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-all \
    exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-all/100-best

  local/get_nbest.sh \
    --num_best 100 \
    --LMWT $LMWT \
    exp/chain_e2e/graph/sb-vocab-train20-varikn.d0.0001-bpe1750/ \
    exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-2020 \
    exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-2020/100-best
fi

if [ $stage -le 23 ]; then
  pushd ../speechbrain_2015-2020-kevat_e2e
  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D-Transcript.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-dev-all-fixed/100-best

  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D-Transcript.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-all/100-best

  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D-Transcript.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-2020/100-best
  popd
fi

if [ $stage -le 24 ]; then
  local/run_rescoring.sh exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-2020/100-best
  local/run_rescoring.sh exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-all//100-best
  local/run_rescoring.sh exp/lfmmi-am/FIX-New-CRDNN-J-contd/2602-1128units//decode_parl-test-2020//100-best
  exit
fi

