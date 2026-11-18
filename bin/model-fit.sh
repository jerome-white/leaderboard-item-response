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
# Each analysis writes into its own directory. diagnostics.csv and
# sampler-diagnostics.csv are each computed once per subject and are
# shared by both, passed into each analysis's sync-report.py call by
# explicit path. sampler-diagnostics.csv holds the standard NUTS
# pathology checks (divergences, BFMI) - cheap, since it reuses the
# same idata diagnostics.py already builds; checking these before
# reading anything into chain-sync/floor-ceiling tells you whether a
# subject's sampler was outright broken (divergent, low BFMI) versus
# merely showing the softer correlated-wandering those two analyses
# look for. sync-report.py takes every input as an explicit path on
# the command line (--diagnostics required; --sampler-diagnostics/
# --chain-sync/--item-info optional, whichever the analysis produced),
# so it never has to assume where a file lives relative to another.
# jsons-to-csv.py then combines each analysis's per-subject sync.json
# files into one CSV per benchmark, one row per subject.
#
# A third step, cross-reference.py, runs after both analyses for a
# subject are done, since it needs a file from each: for every beta
# in that subject's worst-k set, how strongly its chain-dispersion
# correlates with the rest of the worst-k set, next to that item's
# floor/ceiling info. One CSV per benchmark, one row per worst-k beta.

ROOT=`git rev-parse --show-toplevel`

source $ROOT/.venv/bin/activate

export SRC=$ROOT/src/analysis/model-fit
export PYTHONPATH=$ROOT
export MATPLOTLIBRC=$SRC/matplotlibrc

fit() {
    while getopts 'b:i:s:k:w:f:c:' option; do
        case $option in
            b) my_bench=$OPTARG ;;
            i) my_input=$OPTARG ;;
            s) my_subject=$OPTARG ;;
            k) my_worst_k=$OPTARG ;;
            w) my_wk_json=`mktemp --tmpdir=$OPTARG` ;;
            f) my_fc_json=`mktemp --tmpdir=$OPTARG` ;;
            c) my_cr_csv=`mktemp --tmpdir=$OPTARG` ;;
            *)
                echo -e Unrecognized option \"$option\"
                exit 1
                ;;
        esac
    done
    chains="--chains $my_subject/output"

    subject=`realpath --relative-to=$my_input $my_subject`
    output=$my_bench/$subject
    mkdir --parents $output

    diagnostics=$output/diagnostics.csv
    python $SRC/diagnostics.py $chains > $diagnostics

    sampler=$output/sampler-diagnostics.csv
    python $SRC/sampler-diagnostics.py $chains > $sampler

    #
    # worst-k: select the --worst-k worst-converging parameters by
    # R-hat, trace their chains, and check whether they move together.
    #
    out_wk=$output/worst-${my_worst_k}
    selected=$out_wk/selected.csv

    mkdir $out_wk
    python $SRC/select-data.py \
           --worst-k $my_worst_k < $diagnostics > $selected
    python $SRC/trace.py $chains \
           --kind trace \
           --output $out_wk/trace.png < $selected
    python $SRC/chain-sync.py $chains < $selected > $out_wk/chain-sync.csv
    python $SRC/sync-report.py \
           --subject $subject \
           --diagnostics $diagnostics \
           --sampler-diagnostics $sampler \
           --chain-sync $out_wk/chain-sync.csv > $my_wk_json

    #
    # floor-ceiling: per-item raw response rate vs. that item's own
    # parameter's R-hat, across the whole item bank - no worst-k
    # selection involved.
    #
    out_fc=$output/floor-ceiling

    mkdir $out_fc
    python $SRC/item-info.py < $my_subject/stan.json > $out_fc/item-info.csv
    python $SRC/sync-report.py \
           --subject $subject \
           --diagnostics $diagnostics \
           --sampler-diagnostics $sampler \
           --item-info $out_fc/item-info.csv > $my_fc_json

    #
    # cross-reference: for each worst-k beta, how strongly it moves
    # with the rest of the worst-k set, next to that item's info.
    #
    python $SRC/cross-reference.py \
           --subject $subject \
           --chain-sync $out_wk/chain-sync.csv \
           --item-info $out_fc/item-info.csv > $my_cr_csv
}; export -f fit

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

scratch=`mktemp --directory`
trap 'rm --recursive --force $scratch' EXIT

for b in ${_benchmark[@]}; do
    bench=$_output/`basename $b`
    mkdir --parents $bench

    tmp=()
    for i in w f c; do
        tmp+=( -$i `mktemp --directory --tmpdir=$scratch` )
    done

    for s in $b/*; do
        cat <<EOF
fit -b $bench -i $b -s $s -k $_worst_k ${tmp[@]}
EOF
    done | parallel --will-cite --line-buffer

    for ((i=0; i<${#tmp[@]}; i+=2)); do
        case ${tmp[$i]:1} in
            w) out=worst-${_worst_k}-sync ;;
            f) out=floor-ceiling-sync ;;
            c) out=cross-reference ;;
            *) exit 1 ;;
        esac
        cat<<EOF
cat ${tmp[$i+1]}/* | python $SRC/jsons-to-csv.py > $bench/${out}.csv
EOF
    done | parallel --will-cite --line-buffer
done
