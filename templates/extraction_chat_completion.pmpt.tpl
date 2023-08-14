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
    "extracted_text": list[str], // The verbatim text from which the result was extracted. Make sure to escape newlines.
    "rationale": str, // A string containing a natural language explanation for the following answer
    "answer": str // The value of {{term}} extracted from the text. Answer must include units and must be normalized, e.g. (sqr. ft. becomes sq ft)
}

# Examples

Input:
NEW PAGE 11

{{zone_abbreviation}} Zone

CELL (2, 1):
Field
CELL (2, 2):
Value
CELL (3, 1):
{{term}}
CELL (3, 2):
123456 sq ft
CELL (4, 1):
Apartment Area
CELL (4, 2):
10,000


Output:
{
    "extracted_text": ["CELL (3, 2):\n123456 sq ft"],
    "rationale": "The cell that corresponds to the value for {{term}} in this table has this answer.",
    "answer": "123456"
}

Input:
NEW PAGE 32

Section 6.3 Industrial Area & Dimensional Requirements
1 Public Sewer or Public Water
2 Neither Public Sewer nor Public Water
3 For proposed warehouse/manufacturing uses 35%, for all other uses 25%, subject to
Commission's authority to permit greater coverage based on landscape, architectural design
and site development elements that exceed minimum standard of the Regulation
4
Shall be in addition to requirements of 8.1.2 Landscaping and Buffers
6-4

CELL (1, 1): 
CELL (1, 2): 
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 3): 
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 4): 
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 5): 
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 6): 
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 7): 
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 8): 
Table 6.3 Area, Height, Coverage and Setback Requirements
CELL (1, 9): 
CELL (1, 10): 
CELL (2, 1): 
Zone
CELL (2, 2): 
Min
Area
CELL (2, 3): 
Min
Width
CELL (2, 4): 
Min
Depth
CELL (2, 5): 
Max
Stories
CELL (2, 6): 
Max
Height
CELL (2, 7): 
Max % Lot
Coverage
CELL (2, 8): 
Min
Front
Yard
CELL (2, 9): 
Min
Side
Yard
CELL (2, 10): 
Min
Rear
Yard
CELL (3, 1): 
I
CELL (3, 2): 
20,000
sq. ft.
CELL (3, 3): 
100'
CELL (3, 4): 
100'
CELL (3, 5): 
2 1/2
CELL (3, 6): 
50'
CELL (3, 7): 
N/A
CELL (3, 8): 
25'
CELL (3, 9): 
20'
CELL (3, 10): 
20'
CELL (4, 1): 
{{zone_abbreviation}}
CELL (4, 2): 
40,000
sq. ft. 1
CELL (4, 3): 
150'
CELL (4, 4): 
150'
CELL (4, 5): 
2 1/2
CELL (4, 6): 
50'
CELL (4, 7): 
25%³
CELL (4, 8): 
50'
CELL (4, 9): 
20'
CELL (4, 10): 
20'
CELL (5, 1): 
{{zone_abbreviation}}
CELL (5, 2): 
60,000
sq. ft. 2
CELL (5, 3): 
200'
CELL (5, 4): 
200'
CELL (5, 5): 
2 1/2
CELL (5, 6): 
50'
CELL (5, 7): 
25%³
CELL (5, 8): 
50'
CELL (5, 9): 
20'
CELL (5, 10): 
20'4

Output:
{
    "extracted_text": [
        "1 Public Sewer or Public Water",
        "2 Neither Public Sewer nor Public Water",
        "CELL (4, 2): \n40,000\nsq. ft.",
        "CELL (5, 2): \n60,000\nsq. ft."
    ],
    "rationale": "From this page we can infer that the value is conditional on the presence of a public sewer or water system, and there are two different values for the current zone, depending on that.",
    "answer": "40,000 sq ft (if public water or sewer); 60,000 sq ft (otherwise)"
}

Input:
NEW PAGE 66

{{zone_abbreviation}} Zone

{{term}} is 123 sq ft, side length is 10 ft

NEW PAGE 67

{{zone_abbreviation}} Zone - Senior Active Overlay

{{term}} is 1523 sq ft

DKEWKWKDS Zone

{{term}} is 20,000 sq ft

Output:
{
    "extracted_text": ["{{term}} is 123 sq ft"],
    "rationale": "The section titled {{zone_abbreviation}} says the answer explicitly. We ignore sections that specify regulations for overlay districts within this district.",
    "answer": "123 sq ft"
}

Input:
Multi-family building

Output:
null