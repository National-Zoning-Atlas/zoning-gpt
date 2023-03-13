You are an expert information extraction system. You are given a
passage that describes a zoning district and information about
their dimension. Do not fill in values unless the key matches a column in the text. 

Output:
Minimum Lot Size ({{zone_abbreviation}}): value sq ft

You should only output for "District Name" equal to {{zone_abbreviation}}. 

Minimum lot area should be in square feet or in acres. Other synonyms for Lot area are {{synonyms}}.

Make sure every answer is fully justified and do not guess.
If any of these properties are missing, write null.

Input:

{{zone_abbreviation}} Zone

CELL (3, 1):
Lot Size
CELL (3, 2):
40,000 sq ft
CELL (4, 1):
Apartment Area
CELL (4, 2):
10,000 sq ft

Output:

Minimum Lot Size ({{zone_abbreviation}}): 40,000 sq ft

Input:

NEW PAGE 32

{{zone_abbreviation}} Zone

Minimum parcel size is 50,000 sq ft

NEW PAGE 33

R60 Zone

Minimum parcel size is 20,000 sq ft

Output:

Minimum Lot Size ({{zone_abbreviation}}): 50,000 sq ft

Input: {{passage}}

Output: 

