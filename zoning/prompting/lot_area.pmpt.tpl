{% for doc, page in docs%}
* Page number {{page}}
  {{doc}}
{% endfor %}

Please provide the following JSON table. Only fill in properties about {{zone_name}} ({{zone_abbreviation}}). Do not fill in values unless the key matches a column in the text. 

{"District Name":  "name",
"Minimum Lot Area" :  ["area", "cell", "justification"] ,
"Frontage / Front Setback" :  ["distance", "cell"],
"Building Height" :  ["height", "cell"],
}

Make sure every answer is fully justified and do not guess.
If any of these properties are missing, write null.
