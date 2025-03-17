#!/bin/bash

export GIT_ROOT=`git rev-parse --show-toplevel`
source $GIT_ROOT/config.rc || exit 1

_responses=$SCRATCH/var/responses
_questions=$SCRATCH/var/questions
_results=$SCRATCH/opt

while getopts 's:h' option; do
    case $option in
        s) _step=$OPTARG ;;
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

huggingface-cli login --token $HF_BEARER_TOKEN &> /dev/null || exit 1

case $_step in
    1) # Hugging Face download
	src=$GIT_ROOT/src/data
	python $src/list_.py \
	    | python $src/gather_.py \
	    | python $src/reduce_.py --corpus $_responses \
	    | python $src/download_.py \
		     --output $_responses \
		     --question-bank $_questions
	;;
    2) # Stan preparation
	src=$GIT_ROOT/src/model
	tmp=`mktemp`
	script=aggregate-data

	for i in $GIT_ROOT/src/experiments/*.py; do
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
	src=$GIT_ROOT/src/model
	for d in $SCRATCH/opt/*; do
	    echo "[ START `date` ] $d" 1>&2

	    output=$d/output
	    summary=$d/summary.csv
	    mkdir $output 2> /dev/null || rm --recursive --force $output/*
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
	for i in $SCRATCH/opt/*; do
	    if [ -e $i/summary.csv ]; then
		echo "[ `date` ] $i" 1>&2
		split=`basename $i`
		cat <<EOF
python $src/from-stan.py $sample \
       --stan-output $i/output \
       --parameters $i/variables.json \
    | python $src/push-to-hub.py \
	     --split $split \
	     --target $HF_DATASETS_TARGET_
EOF
	    fi
	done | parallel --will-cite --line-buffer
	;;
    *)
	;;
esac

#
# Shutdown
#
if [ $EC2_INSTANCE_ID_ ]; then
    aws ec2 stop-instances --instance-id $EC2_INSTANCE_ID_
fi
