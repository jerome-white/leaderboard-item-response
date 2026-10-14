#!/bin/bash
#
# Submit exactly one AWS Batch job for one benchmark/subject. No
# benchmark/subject selection or looping lives here - fan-out across
# many subjects (e.g. every mmlu/* subject) is the caller's job (find
# | parallel, same pattern as bin/stan-sample.sh), not this script's.
#
# Requires a Batch job queue and job definition to already exist (not
# provisioned by this script) and the irt-stan-sample image (aws/) to
# already be pushed to ECR and referenced by that job definition.

while getopts 's:b:q:d:h' option; do
    case $option in
        s) _subject=$OPTARG ;;
        b) _bucket=$OPTARG ;;
        q) _queue=$OPTARG ;;
        d) _definition=$OPTARG ;;
        h)
            cat <<EOF
Usage: $0
 -s benchmark/subject (required, e.g. mmlu/chemistry)
 -b S3 bucket holding opt/<subject>/{stan.json,init.json} (required)
 -q Batch job queue (required)
 -d Batch job definition (required)
EOF
            exit 0
            ;;
        *)
            echo -e Unrecognized option \"$option\"
            exit 1
            ;;
    esac
done

: "${_subject:?-s is required}"
: "${_bucket:?-b is required}"
: "${_queue:?-q is required}"
: "${_definition:?-d is required}"

name=`echo $_subject | tr '/' '-'`
overrides=`printf '{"environment":[{"name":"S3_BUCKET","value":"%s"},{"name":"SUBJECT_PATH","value":"%s"}]}' "$_bucket" "$_subject"`

aws batch submit-job \
    --job-name "$name" \
    --job-queue "$_queue" \
    --job-definition "$_definition" \
    --container-overrides "$overrides"
