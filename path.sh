export PYTHONIOENCODING='utf-8'
module load kaldi-strawberry/2b1b041-staticmath-gcc8.4.0-cuda11.0.2-openblas0.3.13-openfst1.6.7
module load cuda/11.0.2
module load variKN
module load sentencepiece
source speechbrain/speechbrain-env/bin/activate
export PYTHONPATH=$PYTHONPATH:pychain/

export PATH=$PWD/utils:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD:pychain/openfst-1.7.5/lib/

export LC_ALL=C

