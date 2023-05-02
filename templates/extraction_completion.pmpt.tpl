# Instructions

You are an expert architectural lawyer. You are looking for facts inside a
document about a Zoning District with the name "{{zone_name}}" and with short
code name "{{zone_abbreviation}}".

You are looking to find the value for "{{term}}" which also is called:
{{synonyms}}. Only output values that are seen in the input and do not guess!
Output MUST be valid JSON, and should follow the schema of the following
examples. Ensure that the field "extracted_text" does not span multiple lines.

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
    "pages": [11]
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
    "pages": [32, 33]
}

Input:

Multi-family building

Output:

null

# Test

Input:

{{passage}}

Output: 

