#!/bin/bash

ROOT=`git rev-parse --show-toplevel`

export PYTHONPATH=$ROOT
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
# export NUMEXPR_MAX_THREADS=`nproc`

source $HOME/.keys/hf
source $ROOT/.venv/bin/activate

while getopts 'o:h' option; do
    case $option in
        o) _output=$OPTARG ;;
        h)
            cat <<EOF
Usage: $0
 -o Output
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

src=$ROOT/src/data
database=$_output/metadata/db.sqlite
responses=$_output/responses

python $src/initialize_.py --database $database
python $src/list_.py --exclude-flagged \
    | python $src/gather_.py \
    | python $src/reduce_.py --corpus $responses \
    | python $src/download_.py \
             --output $responses \
             --database $database
