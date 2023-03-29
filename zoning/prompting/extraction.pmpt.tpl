You are an expert architectural lawyer. You are looking for facts about a District Name: {{zone_abbreviation}}.
You are looking to find the value for "{{term}}" which also is called: {{synonyms}}. Only output values that are seen in the input and do not guess!



Input:

{{zone_abbreviation}} Zone

CELL (3, 1):
{{term}}
CELL (3, 2):
123456 sq ft
CELL (4, 1):
Apartment Area
CELL (4, 2):
10,000 sq ft

Output:

* {{term}} ({{zone_abbreviation}}): 123456 sq ft
* Reason: CELL (3, 2): 123456 sq ft

Input:

NEW PAGE 32

{{zone_abbreviation}} Zone

{{term}} is 123 sq ft, side length is 10 ft

NEW PAGE 33

DKEWKWKDS Zone

{{term}} is 20,000 sq ft

Output:

* {{term}} ({{zone_abbreviation}}): 123 sq ft
* Reason: {{term}} is 123 sq ft

Input:

Multi-family building

Output:

N/A

Input:

{{passage}}

Output: 

