This directory contains the main scripts to run to setup the data extraction pipeline.


* assemble.py ->  A script for converting the raw Textract Output to a simple text format. Reads from `xyzNLP/nza-ct-zoning-codes` and writes to `xyzNLP/nza-ct-zoning-codes-text`

* index_towns.py -> A script for converting the textual representations to a search engine (elastic search). Indexes every page for 2000 running tokens.

* search.py -> A script that searches for terms based on the thesaurus and makes it easy to look at and debug search queries

* get_districts.py -> A script to run search queries and extract important terms. 


## Data

* thesaurus.json -> Collection of synonym names for common words.

* districts.jsonl -> Extracted districts from each document
