#!/bin/bash

export GIT_ROOT=`git rev-parse --show-toplevel`
source $GIT_ROOT/config.rc || exit 1

_data=$SCRATCH/var/data
_docs=$SCRATCH/var/documents
_results=$SCRATCH/opt

huggingface-cli login --token $HF_BEARER_TOKEN &> /dev/null || exit 1

#
# Hugging Face download
#

src=$GIT_ROOT/src/data

python $src/list_.py \
    | python $src/gather_.py \
    | python $src/reduce_.py --corpus $_data \
    | python $src/download_.py \
	     --output $_data \
	     --question-bank $_docs

#
# Stan preparation
#

src=$GIT_ROOT/src/model
tmp=`mktemp`
script=aggregate-data

for i in $GIT_ROOT/src/experiments/*.py; do
    python $i --output $_results \
	| while read; do
	out=`dirname $REPLY`

	agg=$out/${script}.csv
	python $src/${script}.py \
	       --data-root $_data \
	       --question-bank $_docs \
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

#
# Stan sampling
#

for d in $SCRATCH/opt/*; do
    echo "[ START `date` ] $d" 1>&2
    $GIT_ROOT/bin/sample.sh -d $d -w 500
done

#
# Hugging Face upload
#

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

#
# Shutdown
#
if [ $EC2_INSTANCE_ID_ ]; then
    aws ec2 stop-instances --instance-id $EC2_INSTANCE_ID_
fi
