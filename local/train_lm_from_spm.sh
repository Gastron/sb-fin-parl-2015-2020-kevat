#!/bin/bash
set -eu
. cmd.sh

stage=1
BPE_units=1750
varikn_scale=0.0001
varikn_cmd="$varikn_cmd"
varikn_extra="--clear_history --3nzer --arpa"
skip_lang=false
traindata=
validdata=

echo $0 $@

. path.sh
. parse_options.sh


if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <sent-piece-model> <lm-data> <outdir>"
	echo "e.g.: $0 --BPE-units 1000 train data/lang_bpe.1000.varikn"
  echo "If you don't have LM data dir with files plain_text and plain_text.valid,"
  echo "you can specify:"
  echo "    --stage 0 --traindata <traindata> --validdata <validdata>"
  echo "to copy preprocessed text files."
  exit 1
fi

modeldir="$1"
lmdata="$2"
outdir="$3"

lmdatadir="data/lm_$lmdata"
lmdir="exp/lm/${lmdata}_varikn.bpe${BPE_units}.d${varikn_scale}"
mkdir -p "$lmdir"
mkdir -p "$lmdatadir"

if [ $stage -le 0 ]; then
  echo "Copying text files"
  cp $traindata $lmdatadir/plain_text
  cp $validdata $lmdatadir/plain_text.valid
fi

if [ "$stage" -le 1 ]; then
  echo "Encoding with SentencePiece BPE: $BPE_units"

  # Vocab to plain vocab ( normal SPM format is <subword> <id> ) 
  cut -f1 "$modeldir"/spm.$BPE_units.vocab > "$lmdir"/bpe.$BPE_units.vocab.plain

  $train_cmd "$lmdatadir"/log/spm_encode_"$BPE_units".log \
    spm_encode --model="$modeldir"/spm."$BPE_units".model \
    --output_format=piece \< "$lmdatadir"/plain_text \> "$lmdatadir"/text.bpe.$BPE_units
  $basic_cmd "$lmdatadir"/log/spm_encode_"$BPE_units"_valid.log \
    spm_encode --model="$modeldir"/spm."$BPE_units".model \
    --output_format=piece \< "$lmdatadir"/plain_text.valid \> "$lmdatadir"/valid.bpe.$BPE_units
fi

if [ "$stage" -le 2 ]; then
  local/train_varikn.sh \
		--cmd "$varikn_cmd"  \
		--scale "$varikn_scale" \
		--extra-args "$varikn_extra" \
    "cat $lmdatadir/text.bpe.$BPE_units" \
    "cat $lmdatadir/valid.bpe.$BPE_units" \
    "$lmdir" 
fi

if [ "$stage" -le 3 ]; then
  echo "Compute perplexity"
  perplexity --arpa "$lmdir"/varikn.lm.gz \
    "$lmdatadir"/valid.bpe."$BPE_units" \
    "$lmdir"/valid_perplexity
fi

if [ "$skip_lang" = true ]; then
	echo "Skipping lang dir creation."
	exit 0
fi

dict_dir="data/local/dict_${lmdata}_bpe.$BPE_units"
if [ "$stage" -le 4 ]; then
	echo "Make SentencePiece LM."
	local/make_spm_lang.sh "$lmdir"/bpe.${BPE_units}.vocab.plain $dict_dir $outdir
fi

if [ "$stage" -le 5 ]; then
	echo "Convert ARPA to FST."
	utils/format_lm.sh \
		$outdir "$lmdir"/varikn.lm.gz \
		$dict_dir/lexicon.txt $outdir
fi

