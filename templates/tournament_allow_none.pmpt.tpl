# Instructions

You are an expert architectural lawyer. You are looking for facts inside a
document about a Zoning District with the name "{{district.full_name}}" and with an
abbreviated name "{{district.short_name}}".

You are looking to find the value for "{{term}}", which also goes by the
following other names: {{synonyms}}. You will be shown a list of page contents
from a document. You will also be shown a possible value for {{term}} that was
extracted from the text, along with the rationale for that extraction. Your job
is output the index of the correct answer, as supported by the text. If neither 
answer is correct, you should return -1. For {{term}} in residential districts, 
we are only interested in the answer as it pertains to single-family homes.

Output MUST be a single integer, the 0-based index that selects the best answer
from the list below. If you think both answers are incorrect, return the integer -1 to 
indicate that there is no answer. Do not provide any explanation.

Here are a few examples for reference
# Examples 

Input:
INDEX 0
Answer: 2 acres
Rationale: The section specifies the minimum lot size for a single-family home.
Extracted Text: ['a. A primary dwelling must be on a plot larger than 2 acres in size.']
Supporting Text:
For dwellings located in {{district.full_name}}:
a. A primary dwelling must be on a plot larger than 2 acres in size. 
b. No more than one single-fmaily home to a plot with the exception of guest homes

INDEX 1
Answer: 30,000 sq. ft.
Rationale: The cell that corresponds to the value for maximum lot coverage in this table has this answer for the Elmwood Commercial (E-M) zone.
Extracted Text: ['CELL (1, 1): \nMaximum lot coverage', 'CELL (1, 2): \n30,000 sq. ft.']
Supporting Text:
Height and area requirements.
CELL (1, 1): 
Maximum lot coverage
CELL (1, 2): 
30,000 sq. ft.
CELL (2, 1): 
Minimum lot width
CELL (2, 2): 
6 acres
CELL (3, 1): 

Output: 
1

Input: 
INDEX 0
Answer: 500 square feet
Rationale: The section discusses the land area requirements in Commercial (C)
Extracted Text: ['The minimum land allotted to a business is set to be 500 sq ft']
Supporting Text: 
The minimum land allotted to a business is set to be 500 sq ft. Though this is a small amount, 
the standard allotment is 20000 sq ft for businesses located in the upper Commercial (C) district. 

Index 1
Answer: 
Rationale: There is no reference to {{term}} for district {{district.full_name}}
Extracted Text: ['Not found']
Supporting Text: 

Output: 1 

Input: 
INDEX 0
Answer: 120 ft
Rationale: The table contains the relevant value.
Extracted Text: ['Cell (3, 2): \n120 ft']
Supporting Text: 
NEW PAGE 11

Dartie Zone

CELL (2, 1):
Field
CELL (2, 2):
Value
CELL (3, 1):
Minimum Building Height
CELL (3, 2):
120 ft

Index 1
Answer: 12 ft
Rationale: The table contains the relevant minimum value.
Extracted Text: ['Cell (5, 2): \n12 ft']
Supporting Text: 
NEW PAGE 11

Commercial Zone

CELL (4, 1):
Maximum Building Height
CELL (4, 2):
35,000 ft
CELL (5, 1):
Minimum Driveway Length
CELL (5, 2):
12 ft

Output: -1

# Answers to choose from

{{answers}}