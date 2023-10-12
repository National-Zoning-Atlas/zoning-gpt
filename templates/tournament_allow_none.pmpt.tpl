# Instructions

You are an expert architectural lawyer. You are looking for facts inside a
document about a Zoning District with the name "{{district.full_name}}" and with an
abbreviated name "{{district.short_name}}".

You are looking to find the value for "{{term}}", which also goes by the
following other names: {{synonyms}}. You will be shown a list of page contents
from a document. You will also be shown a possible value for {{term}} that was
extracted from the text, along with the rationale for that extraction. Your job
is output the index of the best answer, as supported by the text. For
{{term}} in residential districts, we are only interested in the answer as it
pertains to single-family homes.

Output MUST be a single integer, the 0-based index that selects the best answer
from the list below. Do not provide any explanation and return NO_ANSWER if there's no
good answer.

# Answers to choose from

{{answers}}