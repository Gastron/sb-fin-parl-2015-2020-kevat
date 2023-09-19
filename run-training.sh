#!/bin/bash
stage=1



. path.sh
. cmd.sh
. utils/parse_options.sh




if [ $stage -le 10 ]; then
  local/run_training.sh --hparams hyperparams/mtl/FIX-New-CRDNN-J.yaml

fi

if [ $stage -le 11 ]; then
  local/run_training.sh --hparams hyperparams/mtl/FIX-New-CRDNN-J-contd.yaml

fi

if [ $stage -le 12 ]; then
  local/chain-decode.sh \
    --stage 2 --posteriors_from "exp/mtl-am/FIX-New-CRDNN-J/2602-2328units//decode_parl-dev-all-fixed" \
    --graphdir "exp/chain/graph/sb-vocab-train20-varikn.d0.0001-bpe1750/" \
    --hparams hyperparams/mtl/FIX-New-CRDNN-J.yaml \
    --decodedir "exp/mtl-am/FIX-New-CRDNN-J/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-train20-varikn.d0.0001-bpe1750/"
fi

if [ $stage -le 13 ]; then
  local/chain-decode.sh \
    --stage 2 --posteriors_from "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed" \
    --graphdir "exp/chain/graph/sb-vocab-train20-varikn.d0.0001-bpe1750/" \
    --hparams hyperparams/mtl/FIX-New-CRDNN-J-contd.yaml \
    --decodedir "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-train20-varikn.d0.0001-bpe1750/"
fi

if [ $stage -le 13 ]; then
  local/chain-decode.sh \
    --stage 2 --posteriors_from "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all" \
    --datadir data/parl-test-all \
    --graphdir "exp/chain/graph/sb-vocab-train20-varikn.d0.0001-bpe1750/" \
    --hparams hyperparams/mtl/FIX-New-CRDNN-J-contd.yaml \
    --decodedir "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-train20-varikn.d0.0001-bpe1750/"
fi

if [ $stage -le 14 ]; then
  local/chain-decode.sh \
    --stage 2 --posteriors_from "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-2020" \
    --datadir kaldi-s5/data/parl2020-test/ \
    --graphdir "exp/chain/graph/sb-vocab-train20-varikn.d0.0001-bpe1750/" \
    --hparams hyperparams/mtl/FIX-New-CRDNN-J-contd.yaml \
    --decodedir "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-2020_sb-vocab-train20-varikn.d0.0001-bpe1750/"
fi

if [ $stage -le 15 ]; then
  local/chain-decode.sh \
    --stage 2 --posteriors_from "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_yle-test-new" \
    --datadir kaldi-s5/data/yle-test-new \
    --graphdir "exp/chain/graph/sb-vocab-train20-varikn.d0.0001-bpe1750/" \
    --hparams hyperparams/mtl/FIX-New-CRDNN-J-contd.yaml \
    --decodedir "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_yle-test-new_sb-vocab-train20-varikn.d0.0001-bpe1750/"
fi



if [ $stage -le 16 ]; then
  local/chain-decode.sh \
    --stage 2 --posteriors_from "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed" \
    --graphdir "exp/chain/graph/sb-vocab-parl30M-varikn.d0.0001-bpe1750/" \
    --hparams hyperparams/mtl/FIX-New-CRDNN-J-contd.yaml \
    --decodedir "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750"
fi

if [ $stage -le 17 ]; then
  local/chain-decode.sh \
    --stage 2 --posteriors_from "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all" \
    --datadir data/parl-test-all \
    --graphdir "exp/chain/graph/sb-vocab-parl30M-varikn.d0.0001-bpe1750/" \
    --hparams hyperparams/mtl/FIX-New-CRDNN-J-contd.yaml \
    --decodedir "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-parl30M-varikn.d0.0001-bpe1750/"
fi

if [ $stage -le 18 ]; then
  local/chain-decode.sh \
    --stage 2 --posteriors_from "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-2020" \
    --datadir kaldi-s5/data/parl2020-test/ \
    --graphdir "exp/chain/graph/sb-vocab-parl30M-varikn.d0.0001-bpe1750/" \
    --hparams hyperparams/mtl/FIX-New-CRDNN-J-contd.yaml \
    --decodedir "exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-2020_sb-vocab-parl30M-varikn.d0.0001-bpe1750"
fi


LMWT=$(cat exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750/scoring_kaldi/wer_details/lmwt)
if [ $stage -le 19 ]; then
  #echo 8 > exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750/num_jobs
  local/get_nbest.sh \
    --num_best 100 \
    --LMWT $LMWT \
    exp/chain/graph/sb-vocab-parl30M-varikn.d0.0001-bpe1750 \
    exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units/decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750/num \
    exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units/decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best

  #echo 8 > exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-parl30M-varikn.d0.0001-bpe1750/num_jobs
  local/get_nbest.sh \
    --num_best 100 \
    --LMWT $LMWT \
    exp/chain/graph/sb-vocab-parl30M-varikn.d0.0001-bpe1750 \
    exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-parl30M-varikn.d0.0001-bpe1750 \
    exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best

  #echo 8 > exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-2020_sb-vocab-parl30M-varikn.d0.0001-bpe1750/num_jobs
  local/get_nbest.sh \
    --num_best 100 \
    --LMWT $LMWT \
    exp/chain/graph/sb-vocab-parl30M-varikn.d0.0001-bpe1750 \
    exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-2020_sb-vocab-parl30M-varikn.d0.0001-bpe1750 \
    exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-2020_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best
fi

if [ $stage -le 20 ]; then
  pushd ../speechbrain_2015-2020-kevat_e2e
  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best

  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best

  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-2020_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best
  popd
fi

if [ $stage -le 21 ]; then
  local/run_rescoring.sh --lm_name trafo_30M exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best
  local/run_rescoring.sh --lm_name trafo_30M exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best
  local/run_rescoring.sh --lm_name trafo_30M exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-2020_sb-vocab-parl30M-varikn.d0.0001-bpe1750/100-best
fi


LMWT=$(cat exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-train20-varikn.d0.0001-bpe1750/scoring_kaldi/wer_details/lmwt)
if [ $stage -le 22 ]; then
  #echo 8 > exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-train20-varikn.d0.0001-bpe1750/num_jobs
  local/get_nbest.sh \
    --num_best 100 \
    --LMWT $LMWT \
    exp/chain/graph/sb-vocab-train20-varikn.d0.0001-bpe1750/ \
    exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-train20-varikn.d0.0001-bpe1750/ \
    exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-train20-varikn.d0.0001-bpe1750/100-best

  #echo 8 > exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-train20-varikn.d0.0001-bpe1750/num_jobs
  local/get_nbest.sh \
    --num_best 100 \
    --LMWT $LMWT \
    exp/chain/graph/sb-vocab-train20-varikn.d0.0001-bpe1750/ \
    exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-train20-varikn.d0.0001-bpe1750 \
    exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-train20-varikn.d0.0001-bpe1750/100-best

  local/get_nbest.sh \
    --num_best 100 \
    --LMWT $LMWT \
    exp/chain/graph/sb-vocab-train20-varikn.d0.0001-bpe1750/ \
    exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-2020_sb-vocab-train20-varikn.d0.0001-bpe1750 \
    exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-2020_sb-vocab-train20-varikn.d0.0001-bpe1750/100-best
fi

if [ $stage -le 23 ]; then
  pushd ../speechbrain_2015-2020-kevat_e2e
  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D-Transcript.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-train20-varikn.d0.0001-bpe1750/100-best

  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D-Transcript.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-train20-varikn.d0.0001-bpe1750/100-best

  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D-Transcript.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-2020_sb-vocab-train20-varikn.d0.0001-bpe1750/100-best
  popd
fi

if [ $stage -le 24 ]; then
  local/run_rescoring.sh exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-train20-varikn.d0.0001-bpe1750/100-best/
  local/run_rescoring.sh exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-train20-varikn.d0.0001-bpe1750//100-best
  local/run_rescoring.sh exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-2020_sb-vocab-train20-varikn.d0.0001-bpe1750//100-best
  exit
fi



if [ $stage -le 25 ]; then
  pushd ../speechbrain_2015-2020-kevat_e2e
  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-train20-varikn.d0.0001-bpe1750/100-best

  srun --mem 16G --gres=gpu:1 --constraint volta --time 2:0:0 \
    ./rescore-trafo-lm.py hyperparams/lm/Trafo-D.yaml \
    --test_dir ../speechbrain_2015-2020-kevat/exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-train20-varikn.d0.0001-bpe1750/100-best
  popd
fi

if [ $stage -le 26 ]; then
  local/run_rescoring.sh --lm_name trafo_30M exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-dev-all-fixed_sb-vocab-train20-varikn.d0.0001-bpe1750//100-best
  local/run_rescoring.sh --lm_name trafo_30M exp/mtl-am/FIX-New-CRDNN-J-contd/2602-2328units//decode_parl-test-all_sb-vocab-train20-varikn.d0.0001-bpe1750//100-best
fi
