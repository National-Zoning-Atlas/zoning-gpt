# Instructions

You are an expert architectural lawyer. You are looking for facts inside a
document about a Zoning District with the name "{{zone_name}}" and with an
abbreviated name "{{zone_abbreviation}}".

You are looking to find the value for "{{term}}", which also goes by the
following other names: {{synonyms}}. Only output values that are seen in the
input and do not guess! Output MUST be valid JSON, and should follow the schema
detailed below. Ensure that the field "extracted_text" does not span multiple
lines. For {{term}} in residential districts, we are only interested in the
answer as it pertains to single-family homes.

# Schema
{
    "answer": str, // The value of {{term}} extracted from the text. Answer must include units and must be normalized, e.g. (sqr. ft. becomes sq ft)
    "extracted_text": str, // The verbatim text from which the result was extracted. Make sure to escape newlines.
    "pages": list[int], // The pages that were used to generate the result. 
    "confidence": float // The confidence value that you have in your answer. Must be between 0.0 and 1.0, inclusive. 1.0 means you are absolutely certain this is the correct answer, 0.0 means this is certainly the wrong answer. 0.5 would indicate that this answer could be correct, but it could apply to sub-districts, overlay districts, subdivisions, or something else.
}

# Examples

Input:

{{zone_abbreviation}} Zone

CELL (2, 1):
Field
CELL (2, 2):
Value (square feet)
CELL (3, 1):
{{term}}
CELL (3, 2):
123456
CELL (4, 1):
Apartment Area
CELL (4, 2):
10,000

NEW PAGE 11

Output:
{
    "answer": "123456 sq ft",
    "extracted_text": "CELL (3, 2): 123456 sq ft",
    "pages": [11],
    "confidence": 1.0
}

Input:

NEW PAGE 32

{{zone_abbreviation}} Zone

{{term}} is 123 sq ft, side length is 10 ft

NEW PAGE 33

DKEWKWKDS Zone

{{term}} is 20,000 sq ft

Output:
{
    "answer": "123 sq ft",
    "extracted_text": "{{term}} is 123 sq ft",
    "pages": [32, 33],
    "confidence": 1.0
}

Input:

NEW PAGE 66

{{zone_abbreviation}} Zone — Active Senior Overlay

{{term}} is 123 sq ft, side length is 10 ft

NEW PAGE 67

DKEWKWKDS Zone

{{term}} is 20,000 sq ft

Output:
{
    "answer": "123 sq ft",
    "extracted_text": "{{term}} is 123 sq ft",
    "pages": [66, 67],
    "confidence": 0.5
}

Input:

Multi-family building

Output:
null