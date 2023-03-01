{% for doc, page in docs%}
* {{doc}}
{% endfor %}

The previous text contains a list of zoning districts and their abbreviation for a town.
Please find all of the district types, zoning district names, and abbreviations.  
Only output districts that have abbreviations.
Please output the answer only with JSON (no text) in the format [{"T": "district type", "Z": "district abbreviation"}].

