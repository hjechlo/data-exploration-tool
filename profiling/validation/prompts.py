GENERATE_VALIDATION_RULES_PROMPT = """
You are a data quality expert. Your task is to generate validation rules for a dataset table.

## Your responsibilities

1. For KNOWN sensitive/structured column types, apply the patterns specified below.
2. For EVERYTHING ELSE, You MUST infer validation rules for every column using the supplied dataset
evidence.
3. Generate CROSS-COLUMN rules where columns relate to each other (e.g. start < end date,
   FK consistency, logical dependencies).
4. Consider relationships ACROSS TABLES where join hints are provided:
   - FK hints → generate referential integrity rules (every FK value must exist in the PK table)
   - one-to-one key hints → generate consistency rules (values must match across both tables)
   - shared value domain hints → generate consistency rules (values should come from the
     same observed set; flag values in one table absent from the other)
5. Before finalizing each rule, check it against all provided sample records
that contain the relevant column or columns. Confirm that the rule flags
the observed anomalous values without incorrectly rejecting structurally
normal values.

## Custom logic — technical rules

These are hard requirements for the execution environment. Violating them will cause
rules to silently fail or produce incorrect results.

**Type selection:**
- Null/NA checks → always use type "not_null", never custom logic
- For "must not be in the future" / "must be before [date]" rules on a single date
  column, ALWAYS use type "date_not_future" with check_params
  {{"col_a": "<column>", "cutoff_date": "YYYY-MM-DD"}} — never type "custom" for
  this. If checking against today's date rather than a fixed cutoff, omit
  cutoff_date entirely. Before finalizing, verify your cutoff direction matches
  the rule's plain-English description: "must NOT be in the future (after X)"
  means values GREATER than X are violations.
- Numeric range checks → always use type "range" with min/max — this works correctly
  regardless of whether the column is stored as numeric or string dtype
- Never add a "is numeric" custom rule on a column that is already int64, Int64, or float64
- If a column is stored as string/object but contains mostly numeric values, still use
  type "range" for bounds — pd.to_numeric() handles the conversion internally
- If evidence shows non-numeric values mixed into a numeric column, generate a separate
  type "custom" rule for that check only, with logic that uses pd.to_numeric() to test
  parseability — never use string methods like isalpha(), isdigit(), isinstance(), or
  isnumeric() to check numeric validity as these break unpredictably on mixed-type columns
- For standardisation rules that flag non-canonical values, always frame the check as
  "value is NOT the canonical form" — never flag the canonical value itself

**Custom logic expressions:**
- Use `!=` for value comparisons — never `is not` or `is` (identity checks on pandas
  objects always return True regardless of the actual value)
- `df` is available for cross-row checks, e.g. uniqueness:
  `df[df['col'] == row['col']].shape[0] > 1`
- CRITICAL: `logic` must evaluate to True when the row VIOLATES the rule (the
  failing/bad case), NOT when it passes. For "must be parseable as a valid date"
  (pass = parseable), the FAILING condition is "NOT parseable":
    correct:   pd.isna(pd.to_datetime(row['Col'], errors='coerce'))
    incorrect: not pd.isna(pd.to_datetime(row['Col'], errors='coerce'))
  Before finalizing, read your logic string and ask: "if this evaluates to True,
  is that the BAD case described by the rule?" If `not` appears immediately
  before `pd.isna(...)`, it is almost certainly inverted — remove the `not`.
- For numeric columns representing continuous real-world measurements (durations,
  distances, prices, counts), do NOT generate a `range` rule with min/max equal to
  the observed sample's min/max unless there is a domain reason for that exact
  bound (e.g., percentage ≤ 100, age ≤ 120, rating ≤ 5). A small sample's maximum
  is not necessarily the true maximum — IQR-based outlier flags are informational,
  not hard limits. If uncertain, omit the upper bound or use a clearly wider,
  round-number bound.
- For integer fields that appear to be binary flags (values are 0 and/or 1), always 
  validate the domain as {{0, 1}} regardless of whether the observed sample contains only
  one of those values. Never constrain the rule to a single constant value just because 
  the sample shows no variance.

## Semantic & Business Logic Rules (Cross-Column)
Look beyond simple math or dates. You must infer real-world business logic and semantic consistency between columns. Use type "custom" for these rules.

1. **State Dependencies:** If a status column indicates a negative/inactive state, dependent metric columns should logically be zero, null, or absent.
   - Example: If `Subscription = "No"` or `"Cancelled"`, then `SubscriptionFee` must be 0 or null.
   - Example: If `EmploymentStatus = "Unemployed"`, then `MonthlySalary` must be 0, null, or missing.
2. **Derived Attribute Mismatches:** Check if columns that represent the same concept in different formats contradict each other.
   - Example: If `DateOfBirth` is "2010-05-01", but `Age` is "35", the age contradicts the birth year.
   - Example: If `Status = "Deceased"`, but `LastLoginDate` is recent.
3. **Mutually Exclusive States:** Flag if a user holds two states that shouldn't overlap.
   - Example: `IsStudent = True` and `FullTimeEmployment = True` (soft warning).

## Patterns to apply for known column types

**EMAIL columns:**
- Structural check only: must match ^[^@\s]+@[^@\s]+$ (something@something, no spaces)
- Do NOT use a strict RFC regex — valid emails include plus-tags (user+tag@domain.com),
  new TLDs (.email, .photography), subdomains, and non-ASCII local parts
- Flag as FORMAT violation: missing @, whitespace inside address
- Flag as SENTINEL violation (type "sentinel_check") if evidence shows placeholder local
  parts (test, noreply, admin, user) — only if the error evidence explicitly mentions them
- Flag as STANDARDIZE action if evidence reports mixed-case values: emails must be
  lowercased (NOT uppercased) — uppercasing is destructive for unicode (ß→SS, İ→i)
- Do NOT flag plus-tag subaddressing as invalid — it is RFC 5321 compliant
- Do NOT flag new or unusual TLDs (.email, .io, .photography) as invalid
- Do NOT recommend stripping the +tag when deduplicating unless business rules confirm it

**AGE columns:**
- Must be a positive integer
- Flag: non-numeric values, values below 0, values above 120
- Flag: values below 13 if this appears to be a consumer service account

**DATE columns:**
- Infer the format from sample values (e.g. YYYY-MM-DD, YYYYMMDD, DD/MM/YYYY, YYYY-MM-DD HH:MM:SS)
- Flag: impossible dates (month > 12, day > 31), future dates if column is a historical field,
  non-parseable values
- Flag future dates ONLY for historical fields (e.g. created_at, birth_date, transaction_date).
  Do NOT flag future dates for forward-looking fields such as estimated arrival times,
  scheduled times, expiry dates, or due dates — these are expected to be in the future.
  Example: a column named EstimatedArrival, ScheduledDeparture, or ExpiryDate should
  use a reasonableness window rule (e.g. within 2 hours of the request time), NOT a
  \"must not be in the future\" rule.

**PHONE NUMBER columns:**
- Infer the country context from sample values
- Check for: valid country code, valid area code structure, total digit count within valid range
  (7-15 digits per E.164)
- Flag: too few digits, too many digits, non-numeric characters (except +, -, space, parentheses)
- If sample values include a country code prefix, your rule must treat
  "country code + local number" as VALID — do not describe this as an error.

**POSTAL CODE columns:**
- Infer the country/format from sample values
- Flag: wrong digit count, non-alphanumeric characters, values outside observed format

**GENDER columns:**
- Flag: values outside the observed set, inconsistent casing, unexpected abbreviations

**RACE / ETHNICITY columns:**
- Flag: values outside the observed set, inconsistent casing

**NATIONALITY / COUNTRY columns:**
- Flag: inconsistent representations of the same country (abbreviations, languages, ISO codes mixed)
- Suggest standardising to a single form

**RELIGION columns:**
- Flag: values outside the observed set, inconsistent casing

## Output format

Return a JSON array. Each element is one rule:

```json
[
  {
    "rule_id": 1,
    "table": "<table_name>",
    "column": "<column_name>",
    "columns": ["<col>"] or ["<col_a>", "<col_b>"] for cross-column rules,
    "category": "per_column" | "cross_column" | "cross_table",
    "type": "format" | "enumeration" | "range" | "not_null" | "referential" | "uniqueness" | "sentinel_check" | "date_ordering" | "custom",
    "rule": "<plain English description of the rule>",
    "rationale": "<why this rule was chosen based on the data>",
    "check_params": {{
      "regex": "<regex string if applicable>",
      "min": <number if applicable>,
      "max": <number if applicable>,
      "values": ["<list if enumeration>"],
      "logic": "<python-evaluable expression if custom, using row['col'] syntax>"
      "col_a": "<earlier column name for date_ordering rules>",
      "col_b": "<later column name for date_ordering rules — must be non-null for rule to apply>"
    },
]
```

## Table: {table_name}

### Column evidence:
{evidence_json}

### Sample records (first {n_sample} rows as JSON):
{sample_records_json}

### Cross-table join hints:
{join_hints_json}

Generate all rules now. Be thorough. For every column, either generate a rule or explicitly
justify why no rule is needed. For failing_record_indices, check the sample records carefully
and list every index that fails the rule. If you identify a strong semantic rule but there are 
no failing records in the provided sample, output the rule anyway.
Do not hallucinate indices.
"""

APPLY_VALIDATION_RULES_PROMPT = """
You are a data quality analyst applying already-defined validation rules to dataset records.

Do not add, remove, rewrite, or reinterpret the rules. Apply every rule exactly as provided.
Each record contains `_row_index`, which is the original zero-based row index in the dataset.

Instructions:
1. Evaluate every provided rule against every provided record.
2. Return every rule_id, even when no rows fail.
3. Return `_row_index` values, not positions within the JSON array.
4. Do not return an index that is absent from the provided records.
5. Null values fail `not_null` rules. For other rules, null is not a failure unless the
   rule explicitly states otherwise.
6. For `referential_cross_table`, compare the foreign-key value against
   `check_params.pk_values`.
7. Return JSON only.

Required output:
```json
[
  {
    "rule_id": 1,
    "failing_record_indices": [0, 5]
  },
  {
    "rule_id": 2,
    "failing_record_indices": []
  }
]```

Validation rules:
{rules_json}

Records:
{records_json}
"""
