You are an expert information extraction system. You are given a
passage that shows the zoning districts of as town and their
abbreviations. Your Job is to list the zoning districts and these
abbreviations.  Only output districts that have abbreviations.  Please
output the answer only with JSON (no text) in the format:


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

{% macro showdocs(docs) -%}
{% for doc in docs %}
* {{doc}}
{% endfor %}
{% endmacro %}
{{showdocs(docs) | truncate(1500*4)}}

Output: