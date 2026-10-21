#!/bin/bash

ROOT=`git rev-parse --show-toplevel`

export PYTHONPATH=$ROOT
source $ROOT/.venv/bin/activate

_src=$ROOT/src/analysis/model-fit

_worst_k=3
_benchmark=()
while getopts 'k:b:h' option; do
    case $option in
        b) _benchmark+=( $OPTARG );;
        k) _worst_k=$OPTARG ;;
        h)
            cat <<EOF
Usage: $0
 -b Benchmark
 -k Worst k (default $_worst_k)
EOF
            exit 0
            ;;
        *)
            echo -e Unrecognized option \"$option\"
            exit 1
            ;;
    esac
done

for b in ${_benchmark[@]}; do
    chains="--chains $b/output"

    output=$b/plots
    mkdir --parents $output

    cat <<EOF
python $_src/diagnostics.py $chains \
    | python $_src/select-data.py --worst-k $_worst_k --parameter theta \
    | MATPLOTLIBRC=$_src/matplotlibrc python $_src/trace.py $chains \
             --kind trace \
             --output $output/trace.png
EOF
done | parallel --will-cite --line-buffer
