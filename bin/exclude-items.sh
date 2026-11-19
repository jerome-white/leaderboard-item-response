#!/bin/bash

# Reruns the data-prep pipeline (build-ids.py -> to-stan/to-variables/
# to-init.py) against a subject's raw aggregate-data.csv, after first
# dropping a given set of document_id/q_i values via exclude-items.py.
# Produces a fresh, internally-consistent stan.json/variables.json/
# init.json in -o, with the excluded items gone and the remaining
# items renumbered contiguously - exactly what bin/stan-sample.sh needs
# to run a refit excluding those items.

ROOT=`git rev-parse --show-toplevel`

export PYTHONPATH=$ROOT

source $ROOT/.venv/bin/activate

src=$ROOT/src/model

_exclude=()
while getopts 'i:o:e:h' option; do
    case $option in
        i) _input=$OPTARG ;;
        o) _output=$OPTARG ;;
        e) _exclude+=( --exclude $OPTARG ) ;;
        h)
            cat <<EOF
Usage: $0
 -i Input subject directory (must contain aggregate-data.csv.gz)
 -o Output directory (will be created; must not already exist)
 -e Item id (document_id/q_i) to exclude (repeatable)
EOF
            exit 0
            ;;
        *)
            echo -e Unrecognized option \"$option\"
            exit 1
            ;;
    esac
done

if [ ${#_exclude[@]} -eq 0 ]; then
    echo "At least one -e is required" >&2
    exit 1
fi

if [ -d $_output ]; then
    echo "$_output exists. Please remove first" >&2
    exit 1
fi
mkdir --parents $_output

scratch=`mktemp --directory`
trap 'rm --recursive --force $scratch' EXIT

tmp=`mktemp --tmpdir=$scratch`

gzip -dc $_input/aggregate-data.csv.gz \
    | python $src/exclude-items.py ${_exclude[@]} \
    | python $src/build-ids.py > $tmp

for j in stan variables init; do
    cat <<EOF
python $src/to-${j}.py --data-file $tmp > $_output/$j.json
EOF
done | parallel --will-cite --line-buffer
