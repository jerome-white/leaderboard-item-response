#!/bin/bash

ROOT=`git rev-parse --show-toplevel`
HF_SAMPLES=0.5

export PYTHONPATH=$ROOT
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export NUMEXPR_MAX_THREADS=`nproc`

source $HOME/.keys/hf
source $ROOT/.venv/bin/activate

while getopts 's:o:t:h' option; do
    case $option in
        s) _step=$OPTARG ;;
        o) _output=$OPTARG ;;
        t) _hf_target=$OPTARG ;;
        h)
            cat <<EOF
Usage: $0
 -s step
    1: Hugging Face upload
EOF
            exit 0
            ;;
        *)
            echo -e Unrecognized option \"$option\"
            exit 1
            ;;
    esac
done

_responses=$_output/hugging-face/responses
_database=$_output/hugging-face/metadata/db.sqlite
_results=$_output/opt
_src=$ROOT/src

scratch=`mktemp --directory`
trap 'rm --recursive --force $scratch' EXIT

err=$scratch/hf.err
hf auth login --token $HF_BEARER_TOKEN &> $err || {
    cat $err
    exit 1
}

case $_step in
    1) # Hugging Face upload
        for i in $_results/*; do
            if [ -e $i/summary.csv ]; then
                echo "[ `date` ] $i" 1>&2
                split=`basename $i`
                cat <<EOF
python $_src/analysis/from-stan.py \
       --sample $HF_SAMPLES \
       --stan-output $i/output \
       --parameters $i/variables.json \
    | python $_src/index/push-to-hub.py \
             --split $split \
             --target $_hf_target
EOF
            fi
        done | parallel --will-cite --line-buffer
        ;;
    *)
        ;;
esac
