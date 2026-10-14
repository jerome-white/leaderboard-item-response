#!/bin/bash

ROOT=`git rev-parse --show-toplevel`

export PYTHONPATH=$ROOT
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export NUMEXPR_MAX_THREADS=`nproc`

source $HOME/.keys/hf
eval "$(pixi shell-hook --shell bash --manifest-path $ROOT/pyproject.toml)"

while getopts 'r:s:t:h' option; do
    case $option in
        r) _results=$OPTARG ;;
        s) _hf_samples="--sample $OPTARG" ;;
        t) _hf_target=$OPTARG ;;
        h)
            cat <<EOF
Usage: $0
 -r results directory
 -s sample fraction (optional, default: full/unsampled)
 -t Hugging Face upload target
EOF
            exit 0
            ;;
        *)
            echo -e Unrecognized option \"$option\"
            exit 1
            ;;
    esac
done

tmp=`mktemp`
trap 'rm --recursive --force $tmp' EXIT
hf auth login --token $HF_BEARER_TOKEN &> $tmp || {
    cat $tmp
    exit 1
}

src=$ROOT/src
for i in $_results/*; do
    if [ -e $i/summary.csv ]; then
        echo "[ `date` ] $i" 1>&2
        split=`basename $i`
        cat <<EOF
python $src/analysis/from-stan.py $_hf_samples \
       --stan-output $i/output \
       --parameters $i/variables.json \
    | python $src/index/push-to-hub.py \
             --split $split \
             --target $_hf_target
EOF
    fi
done | parallel --will-cite --line-buffer
