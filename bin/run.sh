#!/bin/bash

ROOT=`git rev-parse --show-toplevel`
HF_SAMPLES=0.5
STAN_SAMPLES=1000
STAN_WARMUP=500
STAN_WORKERS=`nproc`
CMDSTAN=

export PYTHONPATH=$ROOT
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export NUMEXPR_MAX_THREADS=`nproc`

source $HOME/.keys/hf

while getopts 's:o:t:h' option; do
    case $option in
        s) _step=$OPTARG ;;
	o) _output=$OPTARG ;;
	t) _hf_target=$OPTARG ;;
        h)
            cat <<EOF
Usage: $0
 -s step
    1: Hugging Face download
    2: Stan preparation
    3: Stan sampling
    4: Hugging Face upload
EOF
            exit 0
            ;;
        *)
            echo -e Unrecognized option \"$option\"
            exit 1
            ;;
    esac
done

_responses=$_output/var/responses
_questions=$_output/var/questions
_results=$_output/opt
_src=$ROOT/src

huggingface-cli login --token $HF_BEARER_TOKEN &> /dev/null || exit 1

case $_step in
    1) # Hugging Face download
        src=$_src/data
        python $src/list_.py --exclude-flagged \
            | python $src/gather_.py \
            | python $src/reduce_.py --corpus $_responses \
            | python $src/download_.py \
                     --output $_responses \
                     --question-bank $_questions
        ;;
    2) # Stan preparation
        src=$_src/model
        tmp=`mktemp`
        script=aggregate-data

        for i in $_src/experiments/*.py; do
            python $i --output $_results \
                | while read; do
                echo "[ `date` ] $REPLY" 1>&2
                out=`dirname $REPLY`

                agg=$out/${script}.csv
                python $src/${script}.py \
                       --data-root $_responses \
                       --question-bank $_questions \
                       --experiment $REPLY > $agg

                python $src/build-ids.py < $agg > $tmp
                for j in stan variables; do
                    cat <<EOF
python $src/to-${j}.py --data-file $tmp > $out/$j.json
EOF
                done | parallel --will-cite --line-buffer

                pigz --best $agg
            done
        done

        rm $tmp
        ;;
    3) # Stan sampling
        src=$_src/model
        find $_results -name 'stan.json' \
            | while read; do
            d=`dirname $REPLY`
            echo "[ START `date` ] $d" 1>&2

            output=$d/output
            mkdir $output 2> /dev/null || continue
            summary=$d/summary.csv
            rm --force $summary

            (cd $CMDSTAN && make --jobs=`nproc` $src/model) || exit 1
            $src/model \
                sample \
                num_samples=$STAN_SAMPLES \
                num_warmup=$STAN_WARMUP \
                num_chains=$STAN_WORKERS \
                data \
                file=$d/stan.json \
                output \
                file=$output/chain.csv \
                num_threads=$STAN_WORKERS \
                && stansummary --csv_filename=$summary $output/*.csv

        done
        ;;
    4) # Hugging Face upload
        for i in $_results/*; do
            if [ -e $i/summary.csv ]; then
                echo "[ `date` ] $i" 1>&2
                split=`basename $i`
                cat <<EOF
python $_src/analysis/from-stan.py \
       --sample $HF_SAMPLES \
       --stan-output $i/output \
       --parameters $i/variables.json \
    | python $_src/index/push-to-hub.py \
             --split $split \
             --target $_hf_target
EOF
            fi
        done | parallel --will-cite --line-buffer
        ;;
    *)
        ;;
esac
