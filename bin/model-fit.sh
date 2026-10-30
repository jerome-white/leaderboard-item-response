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
        subject=`realpath --relative-to=$b $s`
        output=$_output/$bench/$subject
        mkdir --parents $output

        ln --relative --symbolic $s/output $output
        ln --relative --symbolic $s/stan.json $output

        diagnostics=$output/diagnostics.csv
        cat <<EOF
python $_src/diagnostics.py \
       --chains $output/output > $diagnostics \
    && python $_src/select-data.py \
              --worst-k $_worst_k < $diagnostics > $output/selected.csv
EOF
    done
done | parallel --will-cite --line-buffer


find $_output -name 'selected.csv' \
    | while read; do
    output=`dirname $REPLY`
    chains="--chains $output/output"

    cat <<EOF
MATPLOTLIBRC=$_src/matplotlibrc python $_src/trace.py $chains \
             --kind trace \
             --output $output/trace.png < $REPLY
python $_src/chain-sync.py $chains > $output/chain-sync.csv < $REPLY
EOF
done | parallel --will-cite --line-buffer

#
# This relies on everything to this point being run; namely that under
# every $_output/[benchmark] exists:
#
#  * per subject directories
#  * each subject directory contains:
#     - a directory of the chains
#     - the stan.json file
#     - diagnostics.csv from diagnostics.py
#     - chain-sync.csv from chain-sync.py
#
# These all happen as side effects of code prior to this.
#
for b in ${_benchmark[@]}; do
    bench=`basename $b`
    output=$_output/$bench
    python $_src/sync-report.py --results $output > $output/sync.csv
done
