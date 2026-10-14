#!/bin/bash

ROOT=`git rev-parse --show-toplevel`
eval "$(pixi shell-hook --shell bash --manifest-path $ROOT/pyproject.toml)"

_stan_samples=1000
_stan_warmup=$_stan_samples
_stan_jobs=1

while getopts 'r:s:w:p:j:h' option; do
    case $option in
        r) _results=$OPTARG ;;
        s) _stan_samples=$OPTARG ;;
        w) _stan_workers=$OPTARG ;;
        p) _stan_warmup=$OPTARG ;;
        j) _stan_jobs=$OPTARG ;;
        h)
            cat <<EOF
Usage: $0
EOF
            exit 0
            ;;
        *)
            echo -e Unrecognized option \"$option\"
            exit 1
            ;;
    esac
done

workers="--jobs=$_stan_jobs"
if [[ -z $_stan_workers ]]; then
    _stan_workers=$(( $(nproc) / _stan_jobs ))
fi
if [ $_stan_workers -lt 1 ]; then
    _stan_workers=1
fi

model=$ROOT/src/model/model
(cd $CMDSTAN && make --jobs=`nproc` $model) || exit 1

find $_results -name 'stan.json' \
    | while read; do
    dname=`dirname $REPLY`

    summary=$dname/summary.csv
    if [ -e $summary ]; then
        echo "Summary $summary exists; skipping $dname" 1>&2
        continue
    fi

    output=$dname/output
    rm --recursive --force $output
    mkdir --parents $output

    cat <<EOF
echo "[ \`date\` ] START $dname" 1>&2 \
    && $model \
           sample \
           num_samples=$_stan_samples \
           num_warmup=$_stan_warmup \
           num_chains=$_stan_workers \
           data \
           file=$dname/stan.json \
           init=$dname/init.json \
           output \
           file=$output/chain.csv \
           num_threads=$_stan_workers &> $dname/log \
    && $CMDSTAN/bin/stansummary \
           --csv_filename=$summary \
           $output/*.csv >> $dname/log \
    && echo "[ \`date\` ] STOP  $dname" 1>&2
EOF
done | parallel --will-cite --line-buffer $workers
