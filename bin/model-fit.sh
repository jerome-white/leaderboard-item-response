#!/bin/bash

# Separate blocks for each distinct analysis this has grown to cover,
# rather than one pipeline that always runs every stage regardless of
# which question is actually being asked. Each block writes into its
# own directory tree - never sharing a file with another block - and
# calls sync-report.py itself on just that tree; sync-report.py
# doesn't know or assume anything about worst-k, it reports whatever
# the files it's given happen to support (see its own comments), so
# keeping blocks on separate trees is what keeps each report scoped
# to its own analysis. No symlinks, no GNU parallel fan-out via
# heredoc generation - plain sequential loops, some repeated
# path-building (diagnostics.py re-run per block rather than shared).
# Clarity over cleverness/speed; revisit parallelism separately if
# this turns out too slow in practice.

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

for b in ${_benchmark[@]}; do
    bench=`basename $b`

    #
    # Block 1 - worst-k / dispersion-correlation analysis. Original
    # #66 tooling: which parameters converge worst, do their chains
    # wander apart together (shared confound) or independently.
    #
    for s in $b/*; do
        subject=`realpath --relative-to=$b $s`
        output=$_output/$bench/worst-k/$subject
        mkdir --parents $output

        echo "[ `date` ] $bench/$subject - worst-k"

        diagnostics=$output/diagnostics.csv
        python $_src/diagnostics.py --chains $s/output > $diagnostics

        selected=$output/selected.csv
        python $_src/select-data.py --worst-k $_worst_k < $diagnostics > $selected

        MATPLOTLIBRC=$_src/matplotlibrc python $_src/trace.py \
                     --chains $s/output \
                     --kind trace \
                     --output $output/trace.png < $selected

        python $_src/chain-sync.py --chains $s/output < $selected > $output/chain-sync.csv
    done

    echo "[ `date` ] $bench sync report - worst-k"
    python $_src/sync-report.py --results $_output/$bench/worst-k > $_output/$bench/worst-k/sync.csv

    #
    # Block 2 - floor/ceiling analysis. Does an item's own raw
    # response rate (rate*(1-rate), peaking at 0.5, ->0 at floor/
    # ceiling) predict its beta's R-hat, across the whole item bank -
    # no worst-k selection, no chain-sync, involved at all. Its own
    # directory, its own diagnostics.csv (re-run, not shared with
    # block 1) - so sync-report.py here sees only this block's files.
    #
    for s in $b/*; do
        subject=`realpath --relative-to=$b $s`
        output=$_output/$bench/floor-ceiling/$subject
        mkdir --parents $output

        echo "[ `date` ] $bench/$subject - floor/ceiling"

        python $_src/diagnostics.py --chains $s/output > $output/diagnostics.csv
        python $_src/item-info.py < $s/stan.json > $output/item-info.csv
    done

    echo "[ `date` ] $bench sync report - floor/ceiling"
    python $_src/sync-report.py --results $_output/$bench/floor-ceiling > $_output/$bench/floor-ceiling/info.csv
done
