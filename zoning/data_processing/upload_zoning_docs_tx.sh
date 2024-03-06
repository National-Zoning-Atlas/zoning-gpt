#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

bucket="nza-zoning"
prefix="zoning/texas-documents"

aws s3 sync --delete data/texas-documents s3://${bucket}/${prefix}

aws --output json s3api list-objects --bucket "${bucket}" --prefix "${prefix}" | jq '.Contents[].Key' | jq -s > data/texas_documents_s3_manifest.json
