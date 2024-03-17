# Instructions

You are an expert information extraction system. You are given a
document and your Job is to provide all the synonyms for
"{{term}}". 

# Examples 
{
  "min_unit_size": [
    "min unit size",
    "min floor area",
    "min finished floor area",
    "min livable floor area",
    "min building size",
  ]
}

Output MUST be valid JSON, and should follow the schema detailed below.

# Schema
{
    ""{{term}}"": list[str] // The list of all the synonyms found in the document
}