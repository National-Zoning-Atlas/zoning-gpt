# Instructions

You are an expert architectural lawyer. You are looking for facts inside a
document about a Zoning District with the name "{{district.full_name}}" and with an
abbreviated name "{{district.short_name}}".

You have identified a potential value for "{{term}}", which also goes by the following other names: {{synonyms}}. You will be given a possible value for {{term}}. You will also be given a rationale for the answer as well as an extracted chunk of text relevant to the answer. Finally, you will be shown the page contents from the document. Your job is to decide whether or not the value is the correct answer for "{{term}}" as supported by the text. For {{term}} in residential districts, we are only interested in the answer as it pertains to single-family homes. Pay extra attention to the district referenced in the text. The given answer should be related to the "{{district.full_name}}" district only, with "{{district.short_name}}" as the ONLY valid abbreviation. The given answer should be a general value for {{term}} in the district, not a special exception or special case. 

If the answer is correct, you should return "Y". If it is incorrect, you should
return "N".

Output MUST be a single character. Do not provide any explanation.

Here is an example for reference: 
# Example 
Input:

Answer: "123456"
Rationale: "The cell that corresponds to the value for {{term}} in this table has this answer."
Extracted Text: ["CELL (3, 2):\n123456 sq ft"]
Supporting Text:

NEW PAGE 11

R-23 Zone

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

N

Explanation: 
The output should be "N" because the extracted answer is for the R-23 Zone instead of the "{{district.full_name}}" Zone.

# Answer to confirm

{{answer}}