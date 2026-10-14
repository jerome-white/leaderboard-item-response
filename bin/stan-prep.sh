#!/bin/bash

ROOT=`git rev-parse --show-toplevel`

export PYTHONPATH=$ROOT
export NUMEXPR_MAX_THREADS=`nproc`

eval "$(pixi shell-hook --shell bash --manifest-path $ROOT/pyproject.toml)"

_src=$ROOT/src

while getopts 'd:r:b:h' option; do
    case $option in
        d) _data=$OPTARG ;;
        r) _responses=$OPTARG ;;
        b) _database=$OPTARG ;;
        h)
            cat <<EOF
Usage: $0
 -o data root (required) - base directory holding responses/ and metadata/
 -r responses directory (optional, default: <data root>/responses)
 -d database path (optional, default: <data root>/metadata/db.sqlite)
EOF
            exit 0
            ;;
        *)
            echo -e Unrecognized option \"$option\"
            exit 1
            ;;
    esac
done

if [[ -z $_responses ]]; then
    _responses=$_data/responses
fi
if [[ -z $_database ]]; then
    _database=$_data/metadata/db.sqlite
fi

scratch=`mktemp --directory`
trap 'rm --recursive --force $scratch' EXIT

tmp=`mktemp --tmpdir=$scratch`
results=$_data/opt
src=$_src/model
script=aggregate-data

for i in $_src/experiments/*.py; do
    python $i --output $results \
        | while read; do
        echo "[ `date` ] $REPLY" 1>&2
        out=`dirname $REPLY`

        agg=$out/${script}.csv
        python $src/${script}.py \
               --data-root $_responses \
               --database $_database \
               --experiment $REPLY > $agg

        python $src/build-ids.py < $agg > $tmp
        for j in stan variables init; do
            cat <<EOF
python $src/to-${j}.py --data-file $tmp > $out/$j.json
EOF
        done | parallel --will-cite --line-buffer
        pigz --best --force $agg
    done
done
