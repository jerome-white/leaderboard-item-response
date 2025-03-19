#!/bin/bash

export GIT_ROOT=`git rev-parse --show-toplevel`
source $GIT_ROOT/config.rc || exit 1

_responses=$SCRATCH/var/responses
_questions=$SCRATCH/var/questions
_results=$SCRATCH/opt
_aggregate=aggregate-data

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

# huggingface-cli login --token $HF_BEARER_TOKEN &> /dev/null || exit 1

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

	for i in $GIT_ROOT/src/experiments/*.py; do
	    python $i --output $_results \
		| while read; do
		echo "[ `date` ] $REPLY" 1>&2
		out=`dirname $REPLY`

		agg=$out/${_aggregate}.csv
		python $src/${_aggregate}.py \
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
	find $SCRATCH/opt -name "${_aggregate}*" \
	    | while read; do
	    i=`dirname $REPLY`
	    echo "[ START `date` ] $i" 1>&2
	    python $src/to-pymc.py \
		   --data-file $REPLY \
		   --save-metadata $i/metadata.json \
		| python $src/irt-model.py --output $i/samples.nc
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
