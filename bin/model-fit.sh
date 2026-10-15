#!/bin/bash

ROOT=`git rev-parse --show-toplevel`

export PYTHONPATH=$ROOT

source $ROOT/.venv/bin/activate

_src=$ROOT/src/analysis/model-fit

while getopts 'd:r:b:h' option; do
    case $option in
        b) _benchmark=$OPTARG ;;
        h)
            cat <<EOF
Usage: $0
 -b Benchmark
EOF
            exit 0
            ;;
        *)
            echo -e Unrecognized option \"$option\"
            exit 1
            ;;
    esac
done

chains="--chains $ROOT/data/opt/mmlu/computer-science/output"
python $_src/diagnostics.py $chains \
    | python $_src/select-worst.py --top-k 10 \
    | python $_src/trace.py $chains \
	     --kind trace \
	     --output $ROOT/tmp/trace.png
