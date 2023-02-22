#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

bucket="cornell-mfd64"
prefix="zoning/orig-documents"

aws s3 sync --delete data/orig-documents s3://${bucket}/${prefix}

aws --output json s3api list-objects --bucket cornell-mfd64 --prefix "${prefix}" | jq '.Contents[].Key' | jq -s > data/orig_documents_s3_manifest.json
