import re

SQ_FT_FORMS = {
    "sq ft",
    "square feet",
    "sq. ft.",
    "sq. ft",
    "sqft",
    "ft2",
    "ft^2",
    "sf",
    "s.f.",
    "SF",
    "sq.ft.",
    "sq-ft",
    "sq. Ft.",
}

ACRE_FORMS = {"acres", "acre", "acreage", "-acre", "ac"}

FT_FORMS = {
    "ft",
    "feet",
    "ft.",
    "\'"
}

PERCENT_FORMS = {
    "%",
    "percent",
    "per cent"
}

def extract_fraction_decimal(text):
    fraction_pattern = r"\d+\s*\d*\/\d+"
    fractions = re.findall(fraction_pattern, text)
    if fractions:
        fraction = fractions[0]
        if " " in fraction:
            whole, fraction = fraction.split()
            numerator, denominator = map(int, fraction.split("/"))
            decimal_value = int(whole) + numerator / denominator
        else:
            numerator, denominator = map(int, fraction.split("/"))
            decimal_value = numerator / denominator
        return decimal_value
    else:
        return 0.0  # TODO: Is this correct? @ek542

# captures x and y, x or y, x; y, x (y)
# split_regex = re.compile(r"\s*(?: or| and|;)\s+")
split_regex = re.compile(r"(.+?)\s(?:\(|or|and)\s(.+?)|(.*?)\sor\s(.*?)|(.*?)\sand\s(.*?)|(.*?)\s\((.*?)\)")
# parses numbers with decimals and commas 
parsing_regex = re.compile(r"(?:\d{4,}|\d{1,3}(?:,\d{3})*)(?:\.\d+)?")

def parse_token(token: str):
    if "/" in token:
        x_split = token.split("/")
        num = len(x_split) - 1
        yield from (extract_fraction_decimal(token) for i in range(num))
    else:
        res = re.findall(parsing_regex, token)
        yield from (float(a.replace(",", "")) for a in res)

def clean_string_units(input_string):
    res = []
    input_string = str(input_string).lower()
    # Split tokens along any conjunctions, e.g. "35 feet or 2.5 stories" -> ["35 feet", "2.5 stories"]
    tokens = split_regex.split(input_string)
    for token in tokens:
        if token:
            if any(substring in token for substring in SQ_FT_FORMS | FT_FORMS | PERCENT_FORMS):
                res.extend(parse_token(token))
            elif any(substring in token for substring in ACRE_FORMS):
                res.extend(t * 43560 for t in parse_token(token))
            else:
                # The token may be unitless
                # TODO: This is a terrible hack! This will fail for any values
                # that are truly between 0 and 1. This is just to accommodate the
                # fact that some zoning codes express what should be a percentage as
                # a decimal fraction. e.g. 0.20 really means 20%.
                res.extend(t * 100 if t < 1 else t for t in parse_token(token))

    return res

