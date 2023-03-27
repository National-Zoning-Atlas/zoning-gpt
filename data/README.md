# Overview

This directory contains most data used by this project. Some of this data is
tracked and managed by DVC. Files which have a `.dvc` file with the same name as
a sibling are tracked by DVC, as are files that correspond to stages in
`dvc.yaml` at the root of this repository.

# Structure

Note many of these files and directories will not be present unless you have run
`dvc pull` at least once.

* `names_all_towns.json`: A JSON file listing machine-friendly names of all
  towns in our dataset.
* `orig-documents`: Folder containing PDFs of all towns listed in
  `names_all_towns.json`.
* `org_documents_s3_manifest.json`: A manifest of all S3 paths in an S3 bucket
  that is used for generating `textract_dataset`. To find the bucket that these
  paths correspond to, see `params.yaml:extract_text_orig_document_s3_bucket`.
* `textract_dataset`: A dataset of raw results from Textract against the zoning
  documents.
* `parquet_dataset`: Equivalent to `textract_dataset` but stored as compressed
  parquet files and a slightly adjusted schema.
* `hf_dataset`: A local copy of a final HuggingFace dataset representing a
  train/test split of the data in `parquet_dataset`.
