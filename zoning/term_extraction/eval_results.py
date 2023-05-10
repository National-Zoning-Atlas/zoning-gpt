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

ACRE_FORMS = {"acres", "acre", "acreage", "-acre"}


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


def clean_string_units(x):
    res = []
    x = x.lower() if isinstance(x, str) else x
    if any(substring in str(x) for substring in SQ_FT_FORMS):
        if "/" in x:
            x_split = x.split("/")
            num = len(x_split) - 1
            res = [extract_fraction_decimal(x) for i in range(num)]
        else:
            res = re.findall(r"(?:\d{4,}|\d{1,3}(?:,\d{3})*)(?:\.\d+)?", x)
            res = [float(a.replace(",", "")) for a in res]
    if any(substring in str(x) for substring in ACRE_FORMS):
        if "/" in x:
            x_split = x.split("/")
            num = len(x_split) - 1
            res = [extract_fraction_decimal(x) * 43560 for i in range(num)]
        else:
            res = re.findall(r"(?:\d{4,}|\d{1,3}(?:,\d{3})*)(?:\.\d+)?", x)
            res = [float(a.replace(",", "")) * 43560 for a in res]
    return res
