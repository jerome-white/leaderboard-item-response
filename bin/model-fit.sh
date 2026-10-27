#!/bin/bash

ROOT=`git rev-parse --show-toplevel`

export PYTHONPATH=$ROOT
source $ROOT/.venv/bin/activate

_src=$ROOT/src/analysis/model-fit

_worst_k=3
_benchmark=()
while getopts 'k:b:o:h' option; do
    case $option in
        b) _benchmark+=( $OPTARG );;
        k) _worst_k=$OPTARG ;;
        o) _output=$OPTARG ;;
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
    bench=`basename $b`
    for s in $b/*; do
        chains="--chains $s/output"

        subject=`realpath --relative-to=$b $s`
        output=$_output/$bench/$subject
        mkdir --parents $output

        cat <<EOF
python $_src/diagnostics.py $chains \
    | python $_src/select-data.py --worst-k $_worst_k \
    | tee >(MATPLOTLIBRC=$_src/matplotlibrc python $_src/trace.py $chains \
             --kind trace \
             --output $output/trace.png) \
          >(python $_src/chain-sync.py $chains > $output/chain-sync.csv) \
          > /dev/null
EOF
    done
done | parallel --will-cite --line-buffer
