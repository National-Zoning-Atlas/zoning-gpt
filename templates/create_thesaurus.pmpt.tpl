# Instructions

You are an expert information extraction system. You are given a
document and your Job is to provide all the alternative terms for
"{{term}}". 

# Examples 
{
  "min_unit_size": [
    "min unit size",
    "min floor area",
    "min finished floor area",
    "min livable floor area",
    "min building size",
    "min floor area",
    "unit size",
    "min size",
    "floor area",
    "min dwelling unit size",
    "floor area requirements",
    "min total living area",
    "min lot area per dwelling unit",
    "living area requirements",
    "min habitable floor area",
    "living area requirements",
    "min gross floor area",
    "min ground floor area"
  ],
  "max_height": [
    "max building height",
    "max height",
    "area and bulk requirements",
    "dimensional requirements",
    "lot and building requirements",
    "area requirements",
    "height",
    "stories",
    "story"
  ]
}

Output MUST be valid JSON, and should follow the schema detailed below.

# Schema
{
    ""{{term}}"": list[str] // The list of all the synonyms found in the document
}

DO NOT fake an answer or make assumptions. Ensure the terms are there in the text, if not return []