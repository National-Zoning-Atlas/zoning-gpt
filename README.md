# Cornell Tech x National Zoning Atlas: Understanding Zoning Codes with Large Language Models

## Introduction

This repository is the result of a collaboration between the team led by Sara
Bronin at the [National Zoning Atlas](https://www.zoningatlas.org/team) and a
team of researchers under Alexander Rush at [Cornell Tech](https://tech.cornell.edu/).

The National Zoning Atlas (NZA) is working to depict key aspects of zoning codes
in an online, user-friendly map. This currently requires extensive human effort
to manually review zoning codes and extract information for each zoning district
within each jurisdiction in the country.

The goal of this project is to use Large Language Models (LLMs) in conjunction
with other natural language processing (NLP) techniques to automatically extract
structured, relevant information from U.S. Zoning Code documents, so as to help
the NZA team expand the reach of its atlas.

## Setup

This repository contains a number of experiments, as well as an automated
evaluation pipeline. Dependencies are managed using `pip`. (Note that this
repository has been tested only with Python 3.10.) To setup your development
environment, run the following:

```sh
brew install pdm
pdm install
```

## Download Existing Artifacts

To use existing artifacts produced by our team, you will need to obtain access
to our Azure Blob Storage. You will need to obtain a credentialed connection
string to this blob storage, and add it to your local DVC configuration using
the following command:

```sh
pdm run dvc remote modify --local cornell-aap-azure connection_string <YOUR CONNECTION STRING HERE>
```

Once you have the DVC remote added, you can pull all existing pipeline data using:

```sh
pdm run dvc pull
```

## Generate/Update Artifacts

If you do not have access to the Azure Blob storage or if you wish to generate your own
results, you can place any number of PDF documents at `data/orig-documents`.

You will need an OpenAI API key available in your environment.

```sh
export OPENAI_API_KEY=<YOUR API KEY HERE>
```

Your environment will also need to have credentials to an AWS IAM identity with
permissions to use AWS Textract and read/write access to an S3 bucket.

```sh
export AWS_DEFAULT_PROFILE=***
# OR
export AWS_ACCESS_KEY_ID=***
export AWS_SECRET_ACCESS_KEY=***
```

You will need to update `params.yaml` to point to your S3 bucket. This bucket
will be used to store your documents so that Textract can run OCR on them. Chang
the `orig_document_s3_bucket` field to be your bucket name.

Finally, you will need to have an ElasticSearch cluster available at
`localhost:9200`. If you don't have ElasticSearch setup, we provide brief
instructions to run a local cluster
[below](#running-an-elasticsearch-cluster-locally).

Once everything running, you can generate a full set of results using the
following command:

```sh
pdm run dvc repro
```

Depending on how many documents you provide, running this may take quite some
time. Anticipate associated costs with using the AWS Textract and OpenAI APIs.
When processing is complete, evaluation metrics will be available in
`data/results/eval.yaml` and the actual generated responses will be available in
`data/results/eval.csv`.

## Running experiments

To run experiments, we recommend using DVC Experiments, which this repository is
setup for.

To run an experiment that changes hyperparameters, you can run something like:

```python
pdm run dvc exp run -S eval.search-method=elasticsearch -S eval.extraction-method=map -S eval.k=12 evaluate --force
```

This will run evaluation with your hyperparameters set to the desired values and
log the results to DVC's experiments tracker. When you have an experiment that you want to commit, run the following commands:

```sh
pdm run dvc apply <experiment-name> # Apply your experiment results to the working tree
git add --update # Add all tracked files that changed to the git index
git commit -m "<your-commit-message>" # Commit
pdm run dvc push # Push DVC-tracked changes to Azure
git push # Push Git-tracked changes to Github
```

Tracking experiments methodically in this way ensures that we don't lose track
of good or interesting results, and helps with reproducing them down the road.

It also lets us perform data science on experiment results to better understand
trends of results. See `maxdumas/exp_analysis.ipynb` for a basic example of
this.

To output all experiment results as a csv for analysis using Pandas, Polars,
etc. you can run a command like:

```sh
pdm run dvc exp show --csv -A > exps.csv
```

## Architecture

TODO

## Appendix

### Running an ElasticSearch Cluster locally

A Docker Compose setup for running a full ElasticSearch stack with Logstash and
Kibana is provided by the Docker Organization. This is the easiest way to run
ElasticSearch locally, but it requires having Docker available on your machine.

If you have Docker available, you can clone the repository and start the cluster
by running:

```
git clone https://github.com/maxdumas/awesome-compose
cd awesome-compose/elasticsearch-logstash-kibana
docker compose up
```

The initial startup may take some time. 

### Instructions for adding new terms

##### STEP 1: Edit/review thesaurus
- Ensure that the new term is in the thesaurus (ex. “min unit size”)
- Ensure that relevant identifier terms like “max” or “min” are included in
  thesaurus (ex. “min” for “min unit size”)
- Add expected dimensions (and its variations) for the term to thesaurus
  (ex. ["square feet", "sq ft", "sf", "s.f.", "sq. ft.", "SF", "sq. ft",
  "sqft", "sq.ft.”] for “min unit size”)
##### STEP 2: Check `ground_truth.csv` file
- Ensure that the two columns corresponding to the term are in the ground_truth.csv file
    - `f”{term}_gt”`: This is the ground truth answer
    - `f”{term}_page_gt”`: This is the ground truth page where you would expect 
       to find the answer
- Note that evaluation metrics will only be produced if this data is
  present. The model will run and produce results anyway but will not
  produce evaluation metrics if the ground truth columns are missing. 
##### STEP 3: Add term to `eval.py`
- In order to run the model for a term, add the term as string to terms
    list in [zoning/data_processing/eval.py](http://eval.py/) (ex. “min unit
    size”) `terms = ["min lot size", "min unit size"] # update to list of terms
    you want to run`
##### STEP 4: Edit/review prompt 
- This will be under [templates/extraction_chat_completion.pmpt.tpl] for
  gpt-3.5 and gpt-4 or [templates/extraction_completion.pmpt.tpl] for gpt-3
  "text-davinci-003"
- We are using a few-shot prompting approach by providing the model with
  various examples of formats in which it may find the answer. Though the
  generalized prompt should work for all numerical terms, you may edit the
  prompt or add additional examples to it if you believe that certain
  formats are missing.
- The suggested path is to start with the default prompt and edit it while
  debugging if you find multiple instances of a format that is not captured
  by the prompt. It is advised to not add very extreme cases as examples in
  the prompt because if it is just an outlier, it will not affect model
  performance significantly.
##### STEP 5: Run
- Run using command `python -m zoning.data_processing.eval` from root
  directory
##### STEP 6: View results
- View metrics in `eval.yaml` and results in `eval.csv` in the
  [data/results] folder

### Debugging and improvement tips

- When adding a new term, first establish a baseline by following the steps
  above.
- Debugging and improving page recall (search):
    - Note that you can only debug search if you have the data for the ground
      truth page for the term. 
    - The main reason for poor search recall is typically the lack of a fleshed
      out thesaurus. The best way to debug this is to add more terms to the
      thesaurus by manually going over the cases where it does not find the
      correct page and seeing what term is used in that zoning doc.
    - We are currently using a simple elastic search query for search. You may
      want to experiment with faiss indexing (embeddings based search) as it may
      work better for cases where a fleshed out thesaurus is not available.
- Debugging and improving answer accuracy (extraction):
    - In order to improve answer accuracy, it is important to understand the
      types of errors. The different types of errors are:
      - Incorrect answer because correct page not found (this can likely be
        addressed by improving page recall performance)
      - Incorrect answer despite correct page being found
        - Answer in complex table, model returns NaN/incorrect cell (this is a
          common error and we have found that even GPT-4 makes such errors when
          tables are complex and especially when keywords like “Minimum” and
          “Lot Size” are split up into different cells of the table)
            - This can be addressed either by changing the way tables are
              represented in text or by providing examples in the prompt that
              capture such cases
         - Answer is in linked table/appendix that is inaccessible (this is not
           very common)
         - Text-based answer with not very obvious wording (the answer is in the
           text but the keywords used are different from what is in the
           thesaurus)
         - Complex answer with multiple values - the model just returns 1 of the
           others or something completely incorrect
- Other things to look out for:
    - Sometimes (very rarely) the answer may be correct (or somewhat correct)
      but may be marked as wrong by our evaluation pipeline.
        - We have two evaluation workflows:
            - For simple numerical answers, we clean the model output (using
              regex) and convert it to a numerical value for direct comparison
              with expected value
            - For complex answers with multiple values, we pass the model output
              to another prompt that compared the expected and returned answers
        - For the numerical workflow, the evaluation pipeline may sometimes
          (very rarely) mark a correct answer as incorrect if it is not in the
          correct format or has an atypical unit. If this happens, you should
          update the clean_string_units function in
          zoning/prompting/eval_results.py to include the edge case you found.
        - For the complex answer workflow, our method is not perfect, so you may
          want to iterate on the prompt to improve evaluation performance.
