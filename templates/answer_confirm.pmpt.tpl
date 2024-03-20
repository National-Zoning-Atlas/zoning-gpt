# Instructions

You are an expert architectural lawyer. You are looking for facts inside a
document about a Zoning District with the name "{{district.full_name}}" and with an
abbreviated name "{{district.short_name}}".

We have identified a potential value for "{{term}}", which also goes by the following other names: {{synonyms}}. You will be given a possible value for {{term}}. You will also be given a rationale for the answer as well as an extracted chunk of text relevant to the answer. Finally, you will be shown the page contents from the document. Your job is to decide whether or not the value is the correct answer for "{{term}}" as supported by the text. For {{term}} in residential districts, we are only interested in the answer as it pertains to single-family homes. Pay extra attention to the district referenced in the text. The given answer should be related to the "{{district.full_name}}" district only, with "{{district.short_name}}" as the ONLY valid abbreviation. The given answer should be a general value for {{term}} in the district, not a special exception or special case. For {{term}} in residential districts, we are only interested in the answer as it pertains to single-family homes.

However, "{{term}}" is different than {{not_synonyms}}, they are not synonyms. If you see any of these "not synonym words" in the text, the answer is incorrect.

{% if value_range %}

The value for {{term}} should be within the range of {{value_range}}. The first number is the minimum value and the second number is the maximum value.

{% endif %}

Clarify that the "{{term}}" applies specifically to the "{{district.full_name}}" or "{{district.short_name}}", not to the other districts, despite both being mentioned on the same page.

If the document did not specify min or max values, you should return "N".

If you think the answer is correct, you should return "Y". If it is incorrect, you should
return "N". If the answer is not in the text, you should also return "N", DO NOT fake an answer or make assumptions.

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


# END of instructions

# Answer to confirm

{{answer}}

# END of answer to confirm