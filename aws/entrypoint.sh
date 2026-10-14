#!/bin/bash
#
# Runs inside the container. One invocation = one benchmark/subject.
# Everything about WHICH subject and WHERE in S3 comes in via env vars,
# set by the local deploy script's job submission - this script has no
# benchmark-selection logic of its own.

set -euo pipefail

: "${S3_BUCKET:?S3_BUCKET must be set}"
: "${SUBJECT_PATH:?SUBJECT_PATH must be set, e.g. mmlu/chemistry}"
: "${NUM_SAMPLES:=1000}"
: "${NUM_WARMUP:=1000}"
: "${NUM_CHAINS:=4}"

s3_prefix="s3://$S3_BUCKET/opt/$SUBJECT_PATH"
local_dir="/tmp/data/$SUBJECT_PATH"
output="$local_dir/output"
mkdir --parents "$output"

echo "[ $(date) ] syncing inputs from $s3_prefix"
aws s3 cp "$s3_prefix/stan.json" "$local_dir/stan.json"
aws s3 cp "$s3_prefix/init.json" "$local_dir/init.json"

echo "[ $(date) ] sampling"
/app/model \
    sample \
    num_samples=$NUM_SAMPLES \
    num_warmup=$NUM_WARMUP \
    num_chains=$NUM_CHAINS \
    data \
    file="$local_dir/stan.json" \
    init="$local_dir/init.json" \
    output \
    file="$output/chain.csv" \
    num_threads=$NUM_CHAINS \
    &> "$local_dir/log"

/app/stansummary --csv_filename="$local_dir/summary.csv" "$output"/chain_*.csv >> "$local_dir/log"

echo "[ $(date) ] compressing chain output"
xz -T0 -6 "$output"/chain_*.csv

echo "[ $(date) ] syncing results back to $s3_prefix"
aws s3 cp "$local_dir/log" "$s3_prefix/log"
aws s3 cp "$local_dir/summary.csv" "$s3_prefix/summary.csv"
aws s3 cp "$output/" "$s3_prefix/output/" --recursive

echo "[ $(date) ] done"
