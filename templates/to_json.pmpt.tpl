Convert text to JSON.
* Unconditional values should use the `general` key.
* Conditional values should use a special key.
* Numerical values should be associated with their units.

Example:
Text: 1 acre (year-round) and 20,000 sq. ft. (seasonal)
```json
{
    "year-round": {
        "acres": 1
    },
    "seasonal": {
        "sq ft": 20000
    }
}
```

Example:
Text: 120,000 sq ft or 2.75 acres
```json
{
    "general": {
        "sq ft": 10000,
        "acres": 2.75
    }
}
```

Example:
Text: 10000
```json
{
    "general": {
        "none": 10000
    }
}
```

Convert the following to JSON.
Text: {{x}}
