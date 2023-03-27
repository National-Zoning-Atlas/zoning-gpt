import json
import time
from os import makedirs
from os.path import basename, dirname, join, realpath, splitext

import boto3
import yaml
from tqdm.contrib.concurrent import thread_map

DIR = dirname(realpath(__file__))

DATA_ROOT = realpath(join(DIR, "..", "..", "data"))

# JSON file listing the full set of towns for which we expect to have data.
input_orig_document_s3_manifest = join(DATA_ROOT, "orig_documents_s3_manifest.json")

output_textract_dataset_path = join(DATA_ROOT, "textract_dataset")

makedirs(output_textract_dataset_path, exist_ok=True)

with open(join(DIR, "..", "..", "params.yaml")) as f:
    config = yaml.safe_load(f)[splitext(basename(__file__))[0]]

PARAM_INPUT_ORIGN_DOCUMENT_S3_BUCKET = config["orig_document_s3_bucket"]

textract = boto3.client("textract")

def start_job(town_pdf_path: str) -> str:
    """
    Runs Textract's StartDocumentAnalysis action and
    specifies an s3 bucket to dump output
    """
    response = textract.start_document_analysis(
        DocumentLocation={"S3Object": {"Bucket": PARAM_INPUT_ORIGN_DOCUMENT_S3_BUCKET, "Name": town_pdf_path}},
        FeatureTypes=[
            "TABLES"
        ],  # TABLES + FORMS is much more expensive ($0.065 per page instead of 0.015)
    )

    return response["JobId"]


def get_job_status(jobId: str):
    """'
    Checks whether document analysis still in progress
    """
    status: str = "IN_PROGRESS"
    while status == "IN_PROGRESS":
        time.sleep(5)
        response = textract.get_document_analysis(JobId=jobId)
        status = response["JobStatus"]
        yield status, response.get("StatusMessage", None)


def get_job_results(jobId: str):
    """
    If document analysis complete, runs Textract's GetDocumentAnalysis action
    and pulls JSON results to be stored in s3 bucket designated above
    """
    response = textract.get_document_analysis(JobId=jobId)
    nextToken = response.get("NextToken", None)
    yield response 

    while nextToken is not None:
        response = textract.get_document_analysis(JobId=jobId, NextToken=nextToken)
        nextToken = response.get("NextToken", None)
        yield response


def run_textract(town_pdf_path: str):
    jobId = start_job(town_pdf_path)
    for s in get_job_status(jobId):
        status, status_message = s
        if status == "FAILED":
            print(f"Job {jobId} on file {town_pdf_path} FAILED. Reason: {status_message}")
        elif status == "SUCCEEDED":
            result = list(get_job_results(jobId))
            with open(join(output_textract_dataset_path, basename(town_pdf_path.replace(".pdf", ".json"))), 'w') as f:
                json.dump(result, f)
            print(f"Job {jobId} on file {town_pdf_path} SUCCEEDED.")

if __name__ == "__main__":
    with open(input_orig_document_s3_manifest) as f:
        all_pdfs = json.load(f)

    thread_map(run_textract, all_pdfs)
