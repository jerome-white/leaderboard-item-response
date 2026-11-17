#!/bin/bash

# Runs two independent analyses per subject (a subject is one model
# fit - one benchmark/topic combination):
#
#   worst-k        Of the parameters that converged worst, do their
#                  chains wander apart together (a shared confound) or
#                  independently?
#   floor-ceiling  Does an item's raw response rate - near 0 or 1 means
#                  little information, near 0.5 means a lot - predict
#                  how well its own parameter converged?
#
# Each analysis writes into its own directory. diagnostics.csv is
# computed once per subject and is the one file both share, passed
# into each analysis's sync-report.py call by explicit path.
# sync-report.py takes every input as an explicit path on the command
# line (--diagnostics required; --chain-sync/--item-info optional,
# whichever the analysis produced), so it never has to assume where a
# file lives relative to another.

ROOT=`git rev-parse --show-toplevel`
SRC=$ROOT/src/analysis/model-fit

source $ROOT/.venv/bin/activate

export PYTHONPATH=$ROOT
export MATPLOTLIBRC=$SRC/matplotlibrc

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
 -b Benchmark (repeatable)
 -k Worst k (default $_worst_k)
 -o Output root
EOF
            exit 0
            ;;
        *)
            echo -e Unrecognized option \"$option\"
            exit 1
            ;;
    esac
done

if [ -d $_output ]; then
    echo "$_output exists. Please remove first" >&2
    exit 1
fi

for b in ${_benchmark[@]}; do
    bench=$_output/`basename $b`

    for s in $b/*; do
        chains="--chains $s/output"

        subject=`realpath --relative-to=$b $s`
        output=$bench/$subject
        diagnostics=$output/diagnostics.csv

        mkdir --parents $output
        python $SRC/diagnostics.py $chains > $diagnostics

        #
        # worst-k: select the --worst-k worst-converging parameters by
        # R-hat, trace their chains, and check whether they move
        # together.
        #
        out_wk=$output/worst-k
        selected=$out_wk/selected.csv
        sync=$out_wk/sync.json

        mkdir $out_wk
        python $SRC/select-data.py \
               --worst-k $_worst_k < $diagnostics > $selected
        python $SRC/trace.py $chains \
               --kind trace \
               --output $out_wk/trace.png \
               < $selected
        python $SRC/chain-sync.py $chains < $selected > $out_wk/chain-sync.csv
        python $SRC/sync-report.py \
               --subject $subject \
               --diagnostics $diagnostics \
               --chain-sync $out_wk/chain-sync.csv \
               > $sync

        #
        # floor-ceiling: per-item raw response rate vs. that item's
        # own parameter's R-hat, across the whole item bank - no
        # worst-k selection involved.
        #
        out_fc=$output/floor-ceiling
        sync=$out_fc/sync.json

        mkdir $out_fc
        python $SRC/item-info.py < $s/stan.json > $out_fc/item-info.csv
        python $SRC/sync-report.py \
               --subject $subject \
               --diagnostics $diagnostics \
               --item-info $out_fc/item-info.csv \
               > $sync
    done
done
