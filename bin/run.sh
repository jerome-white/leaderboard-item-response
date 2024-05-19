#!/bin/bash

ROOT=`git rev-parse --show-toplevel`

export PYTHONPATH=$ROOT
export PYTHONLOGLEVEL=info
export HF_TOKEN=
export NUMEXPR_MAX_THREADS=`nproc`
export TMPDIR=/tmp/huggingface

#
#
#
_s3_bucket=
_s3_path=`date +%F-%H%M%S-%Z`
_ec2_instance=
_flagged=https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard/raw/main/src/leaderboard/filter_models.py

#
#
#
python $ROOT/list-evals.py \
    | python $ROOT/remove-flagged.py --source $_flagged \
    | python $ROOT/list-results.py \
    | python $ROOT/extract-evals.py \
    | python $ROOT/store-evals.py --output s3://$_s3_bucket/$_s3_path

#
#
#
aws ec2 stop-instances --instance-id $_ec2_instance
