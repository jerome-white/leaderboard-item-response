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
        # Block 1 - worst-k / dispersion-correlation
        # analysis. Original #66 tooling: which parameters converge
        # worst, do their chains wander apart together (shared
        # confound) or independently.
        #
        out_wk=$output/worst-k
        selected=$out_wk/selected.csv
        sync=$out_wk/sync.csv

        mkdir $out_wk
        python $SRC/select-data.py \
               --worst-k $_worst_k < $diagnostics > $selected
        python $SRC/trace.py $chains \
               --kind trace \
               --output $out_wk/trace.png \
               < $selected
        python $SRC/chain-sync.py $chains < $selected > $out_wk/chain-sync.csv
        if [ ! -e $sync ]; then
            python $SRC/sync-report.py --results $out_wk > $sync
        fi

        #
        # floor/ceiling analysis. Does an item's own raw response rate
        # (rate*(1-rate), peaking at 0.5, ->0 at floor/ ceiling)
        # predict its beta's R-hat, across the whole item bank - no
        # worst-k selection, no chain-sync, involved at all. Its own
        # directory, its own diagnostics.csv (re-run, not shared with
        # block 1) - so sync-report.py here sees only this block's
        # files.
        #
        out_fc=$output/floor-ceiling
        sync=$out_fc/sync.csv

        mkdir $out_fc
        python $SRC/item-info.py < $s/stan.json > $out_fc/item-info.csv
        if [ ! -e $sync ]; then
            python $SRC/sync-report.py --results $out_fc > $sync
        fi
    done
done
