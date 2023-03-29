You are an expert information extraction system. You are given a
passage that shows the zoning districts of as town and their
abbreviations. Your Job is to list the zoning districts and these
abbreviations.  Only output districts that have abbreviations. 
Please output the answer only with JSON (no text) in the format:

[{"T": "district type", "Z": "district abbreviation with number"}].


Passage:

Some text about buildings

Output:

[]

Passage:

* Residential (R) districts

CELL
Residential
CELL
R-10
CELL
R-20
CELL

Output:

[{"T": "Residential",  "Z": "R-10"}, {"T": "Residential",  "Z": "R-20"}]

Passage:

* Business (C) districts:

(C19) Commercial 19
(C29) Commercial 29

Output:

[{"T": "Commercial 19",  "Z": "C19"}, {"T": "Commercial 29",  "Z": "C29"}]

Passage:

CELL
Residential Districts
CELL
R-5 District
R-10 District
R-20 District

Output:

[{"T": "R-5 Residential",  "Z": "R-5"}, {"T": "R-10 Residential",  "Z": "R-10"}, {"T": "R-20 Residential",  "Z": "R-20"}]

Passage:

Residence AAA District
Residence B District
Historic Design District (HDD)

Output:

[{"T": "Residence AAA",  "Z": "AAA"}, {"T": "Residence B",  "Z": "B"}, {"T": "Historic Design",  "Z": "HDD"}]

Passage: 

{% macro showdocs(docs) -%}
{% for doc in docs %}
* {{doc}}
{% endfor %}
{% endmacro %}
{{showdocs(docs) | truncate(1200*4)}}

Output: