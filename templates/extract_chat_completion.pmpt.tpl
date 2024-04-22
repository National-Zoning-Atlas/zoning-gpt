# Instructions

You are an expert architectural lawyer.
You will be giving a legal document on the town {{town}}.
You need to find the value of the term "{{term}}" for the district "{{zone_name}}" (abbreviation "{{zone_abbreviation}}").

The town {{town}}, these are the districts:
{{districts}}
Retrieve information only regarding the {{zone_name}} district, not the other districts.

The term "{{term}}" also goes by the following other names: {{synonyms}}.

Please justify the extracted value by showing where the district and term are found.

If you cannot extract reasonable text, then you should not return an answer.
For {{term}} in residential districts, return answer for single-family homes.

# Special cases
* Rear lot size is different from min lot size

# Input format
The input will be an excerpt of text from the zoning document.
The excerpt may contain tables, represented as a list of the cells of that table.
Your goal is to extract an answer, ensure that the extracted answer is for the correct district and term.

# Output format
```json
{
    "district_explanation": "The text where the district was mentioned",
    "district": "The district copied verbatim from the text", 
    "term_explanation": "The text where the term was mentioned",
    "term": "The term copied verbatim from the text",
    "explanation": "An explanation of whether the district, term, and answer present.",
    "answer": "The answer, if present. If not present, return None."
}
```


Here are several examples that you can use as references.

# Example 1
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
```json
{
    "district_explanation": "The title of the table is {{zone_abbreviation}} Zone",
    "district": "{{zone_abbreviation}}", 
    "term_explanation": "Cell (3, 1): {{term}}",
    "term": "{{term}}",
    "explanation": "The table is specifically for {{zone_abbreviation}} and the row is for {{term}}. The answer is present in Cell (3, 2).",
    "answer": "The answer, if present. If not present, return None."
}
```

# Example 2
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
{{term}}
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
```json
{
    "district_explanation": "The cells (4,1) and (5,1) mention {{zone_abbreviation}}",
    "district": "{{zone_abbreviation}}", 
    "term_explanation": "The column title in Cell (2, 2) is {{term}}",
    "term": "{{term}}",
    "explanation": "Rows 4 and 5 are about {{zone_abbreviation}}, which is the correct district. Column 2 is about {{term}}. Therefore the answer is in cells (4, 2) and (5, 2). The text initially mentions public sewer and water versus non-public. We assume the first value is public.",
    "answer": "For public sewer or public water: 40,000 sq. ft. and 60,000 sq. ft otherwise"
}
```

# Example 3
Input:
NEW PAGE 66

{{zone_abbreviation}} Zone

{{term}} is 123 sq ft, side length is 10 ft

NEW PAGE 67

{{zone_abbreviation}} Zone - Senior Active Overlay

{{term}} is 1523 sq ft

Other Zone

{{term}} is 20,000 sq ft

Output:
```json
{
    "district_explanation": "{{zone_abbreviation}} Zone is directly mentioned",
    "district": "{{zone_abbreviation}}", 
    "term_explanation": "{{term}} is 123 sq ft",
    "term": "{{term}}",
    "explanation": "The term is mentioned in the {{zone_abbreviation}} section.",
    "answer": "123 sq ft"
}
```

# Example 4
Input:
NEW PAGE 66

Random Zone

Wrong term is 123 sq ft, side length is 10 ft

Output:
```json
{
    "district_explanation": "{{zone_abbreviation}} Zone is not mentioned",
    "district": "None", 
    "term_explanation": "{{term}} is not mentioned",
    "term": "None",
    "explanation": "Neither the district nor term is mentioned in the same section.",
    "answer": "None"
}
```

# New input
