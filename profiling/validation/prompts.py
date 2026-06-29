GENERATE_VALIDATION_RULES_PROMPT = """
You are a data quality expert generating validation rules for a dataset table.
 
## Output format
 
Return a JSON array. Each element is one rule:
 
```json
[
  {{
    "rule_id": 1,
    "table": "<table_name>",
    "column": "<column_name>",
    "columns": ["<col>"],
    "category": "per_column" | "cross_column" | "cross_table",
    "type": "format" | "enumeration" | "range" | "not_null" |
            "referential" | "referential_cross_table" | "uniqueness" |
            "sentinel_check" | "date_ordering" | "date_not_future" |
            "numeric_parseable" | "integer_parseable" |
            "phone_validity" | "custom" | "cross_table_semantic",
    "rule": "<plain-English description>",
    "rationale": "<why this rule was chosen based on the evidence>",
    "check_params": {{
      "regex":            "<regex string, if applicable>",
      "expected_length":  "<integer, if applicable>",
      "min":              "<number, if applicable>",
      "max":              "<number, if applicable>",
      "values":           ["<allowed values for enumeration rules>"],
      "sentinel_values":  ["<explicit sentinel values for sentinel_check rules>"],
      "logic":            "<Python expression using row['col']; True = violation>",
      "col_a":            "<first/primary column, if applicable>",
      "col_b":            "<second/comparison column, if applicable>",
      "ref_table":        "<referenced table for referential rules>",
      "ref_column":       "<referenced column for referential rules>",
      "country_code":     "<country code for phone rules, if applicable>",
      "valid_first_digits": ["<allowed first digits, if applicable>"],
      "dominant_length":  "<expected local phone-number length, if applicable>"
    }}
  }}
]
```
 
---
## Your responsibilities
 
1. For **known sensitive/structured column types**, apply the patterns specified in the
   "Per-column rules for known types" section below.
2. For **everything else**, infer validation rules for every column using the supplied
   evidence. For every column, either generate at least one rule or explicitly justify
   why none is needed.
3. Generate **cross-column rules** where columns relate to each other (e.g. start < end
   date, FK consistency, logical dependencies).
4. Generate **cross-table rules** where join hints are provided:
   - FK hints → referential integrity (every FK value must exist in the PK table).
   - One-to-one key hints → consistency (values must match across both tables).
   - Shared value domain hints → consistency (values should come from the same observed
     set; flag values in one table absent from the other).
   - Semantic cross-column rules also apply across tables. If a column in
     table A and a column in table B are related via a join key **or** carry logically
     dependent meaning (e.g. a name particle encoding gender vs. a gender field), generate 
     a cross-table custom rule scoped to the table containing the primary column.
5. **Verify against sample records** before finalising each rule. Confirm it flags the
   observed anomalous values without incorrectly rejecting structurally normal values.
 
---
## Rule type selection
 
Use this table to pick the correct type. Using `custom` where a named type exists will
cause the rule to fail silently.
 
| Scenario                                        | Use type            | Never use      |
|-------------------------------------------------|---------------------|----------------|
| Null / missing value check                      | `not_null`          | `custom`       |
| Numeric bounds (min / max)                      | `range`             | `custom`       |
| Single-date "must not be in the future" check   | `date_not_future`   | `custom`       |
| Allowed value set                               | `enumeration`       | `custom`       |
| Sentinel / placeholder value detection          | `sentinel_check`    | `custom`       |
| All other checks                                | `custom`            | —              |
 
**`not_null` rules** — generate one only when BOTH conditions are met:
- The column is a mandatory business field (identifier, primary key, required FK, date),
  OR the evidence explicitly shows `missing_pct > 0`.
- A not_null rule that can never fire (no missing values, optional column) adds no value.
 
**`range` rules** — use for numeric bounds regardless of whether the column is stored
as numeric or string dtype (`pd.to_numeric()` handles conversion internally). Do NOT
generate a range rule with `min`/`max` equal to the observed sample's min/max unless a
real domain constraint applies (e.g. percentage ≤ 100, age ≤ 120, rating ≤ 5). If no
domain constraint applies, omit the range rule.
 
**`date_not_future`** — supply `check_params` as:
`{{"col_a": "<column>", "cutoff_date": "YYYY-MM-DD"}}`.
Omit `cutoff_date` when checking against today's date. Before finalising, verify the
cutoff direction: "must NOT be after X" means values **greater than** X are violations.
 
**`sentinel_check`** — `check_params.sentinel_values` is mandatory. Include only values
supported by the supplied evidence. Never generate a sentinel_check rule without it.
 
**`is numeric` custom rules** — never add one for a column already typed int64, Int64,
or float64. If non-numeric values are mixed into a numeric column, generate a separate
`custom` rule using `pd.to_numeric()` to test parseability.
 
**Binary flag columns** (values 0/1) — always validate the domain as {{0, 1}} even if
the sample shows only one value.
 
---

## Custom logic expressions
 
These rules apply to every `"logic"` string in `check_params`.
 
**True = violation.** The expression must evaluate to `True` for rows you want to flag
(the bad case), and `False` for valid rows. Test by reading the expression aloud: "if
this is True, is that the bad case?" If `not` appears immediately before `pd.isna(...)`,
the logic is almost certainly inverted — remove the `not`.
 
```python
# Correct  — True when the value is NOT a parseable date (bad case)
pd.isna(pd.to_datetime(row['Col'], errors='coerce'))
 
# Incorrect — True when the value IS parseable (good case)
not pd.isna(pd.to_datetime(row['Col'], errors='coerce'))
```
 
**Comparison operators** — always use `!=` / `==`, never `is not` / `is`. Identity
checks on pandas objects return True regardless of value.
 
**Cross-row checks** — `df` is available:
```python
df[df['col'] == row['col']].shape[0] > 1   # uniqueness example
```
 
**String methods** — never use `isalpha()`, `isdigit()`, `isinstance()`, or
`isnumeric()` to validate numeric columns. These break on mixed-type columns. Use
`pd.to_numeric(row['col'], errors='coerce')` instead.
 
**Standardisation rules** — frame the check as "value is NOT the canonical form". Never
flag the canonical value itself.

**Cross-table semantic rules** — when a rule requires looking up a value from another
table, use type `"cross_table_semantic"` instead of `"custom"`. Set `"logic": ""` and
populate these fields in `check_params`:
- `"sibling_table"`: the exact name of the other table
- `"join_col"`: the column in the current table used to join
- `"sibling_join_col"`: the matching column in the sibling table
- `"sibling_data_col"`: the column in the sibling table containing the data to check
- `"semantic_check"`: plain-English description of what to check once the lookup is available
Do not reference external dataframes in logic expressions — these will always fail.
Any expression referencing a variable other than `row`, `df`, `pd`, or `re` will fail at runtime.
 
---

## Per-column rules for known types
 
Each entry follows the same structure: **Flag**, **Sentinel**, **Standardise**, **Never**.
 
**EMAIL**
- Flag (format): must match `^[^@\s]+@[^@\s]+$` — flag missing `@` or whitespace inside
  the address.
- Flag (sentinel): type `sentinel_check` if evidence shows placeholder local parts
  (`test`, `noreply`, `admin`, `user`) — only when the error evidence explicitly mentions
  them.
- Standardise: flag mixed-case values if evidence reports them; emails must be
  **lowercased** (not uppercased — uppercasing is destructive for unicode).
- Never: flag plus-tag subaddressing (`user+tag@domain.com`), new or unusual TLDs
  (`.email`, `.io`, `.photography`), or recommend stripping `+tag` for deduplication
  unless business rules confirm it. Do not use a strict RFC regex.
 
**AGE**
- Flag: non-numeric values; values below 0 or above 120.
- Flag: values below 13 if the dataset appears to be a consumer service.
- Sentinel / Standardise: n/a.
- Never: generate a range rule with bounds taken directly from the observed sample.
 
**DATE**
- Flag (format): infer the expected format from sample values (e.g. `YYYY-MM-DD`,
  `DD/MM/YYYY`, `YYYY-MM-DD HH:MM:SS`). Flag non-parseable values and impossible dates
  (month > 12, day > 31).
- Flag (future): use `date_not_future` only for **historical** fields (created_at,
  birth_date, transaction_date). Do NOT flag future dates for forward-looking fields
  (estimated arrival, scheduled departure, expiry date, due date).
- Sentinel / Standardise: n/a.
- Never: apply a "must not be in the future" rule to a forward-looking column.
 
**PHONE NUMBER**
- Flag: too few or too many digits (valid range: 7–15 per E.164); non-numeric characters
  other than `+`, `-`, space, or parentheses.
- Flag: invalid country code or area code structure, inferred from sample values.
- Sentinel / Standardise: n/a.
- Never: flag a number that includes a valid country code prefix as an error — treat
  "country code + local number" as valid.
 
**POSTAL CODE**
- Flag: wrong digit/character count; non-alphanumeric characters; values outside the
  observed format (infer country/format from sample values).
- Sentinel / Standardise / Never: follow evidence.
 
**GENDER**
- Flag: values outside the observed set; inconsistent casing; unexpected abbreviations.
- Sentinel / Standardise / Never: follow evidence.
 
**RACE / ETHNICITY**
- Flag: values outside the observed set; inconsistent casing.
- Sentinel / Standardise / Never: follow evidence.
 
**NATIONALITY / COUNTRY**
- Flag: inconsistent representations of the same country (abbreviations, languages, ISO
  codes mixed).
- Standardise: suggest standardising to a single form.
- Never: follow evidence.
 
**RELIGION**
- Flag: values outside the observed set; inconsistent casing.
- Sentinel / Standardise / Never: follow evidence.
 
---

## Cross-column semantic & business logic rules
 
Use type `"custom"` for all rules in this section. Generate these even if no failing
records appear in the sample — output the rule anyway if the business logic is sound.
 
**1. State dependencies**
If a status column indicates an inactive/negative state, dependent metric columns should
be zero, null, or absent.
- Example: `Subscription = "No"` or `"Cancelled"` → `SubscriptionFee` must be 0 or null.
- Example: `EmploymentStatus = "Unemployed"` → `MonthlySalary` must be 0, null, or
  missing.
 
**2. Derived attribute mismatches**
Columns representing the same concept in different forms must not contradict each other.
- Example: `DateOfBirth = "2010-05-01"` but `Age = 35` — contradicts the birth year.
- Example: `Status = "Deceased"` but `LastLoginDate` is recent.
 
**3. Mutually exclusive states**
Flag records where two states that should not co-exist are both present.
- Example: `IsStudent = True` and `FullTimeEmployment = True`.
 
**4. Conditional presence (field co-dependencies)**
If one field being populated implies another must also be populated, flag records where
the dependent field is absent.
- Example: `street_number` is provided but `street_name` or `address_line_1` is null.
- Example: `discount_code` is present but `discount_amount` is null.
- Scan all address, contact, and composite-key columns for such dependencies.
 
**5. Geographic / administrative consistency**
Where the dataset contains region-specific structured data, validate internal consistency
between geographic fields.
- Australia example: if `country = "Australia"`, validate that `state` belongs to
  `{{NSW, VIC, QLD, SA, WA, TAS, NT, ACT}}` and that `postcode` falls within the known
  range for that state (NSW 1000–2599 **or** NSW 2640–2999, VIC 3000–3999, QLD 4000–4999,
  SA 5000–5999, WA 6000–6999, TAS 7000–7999, NT 0800–0999, ACT 0200–0299 **or** ACT 2600–2639).
- Apply equivalent rules for other countries if evidence supports it (e.g. Singapore
  6-digit postcodes starting with valid district prefixes).
 
**6. Cultural / linguistic name consistency**
Where names contain culturally specific particles that encode gender or relationship, 
validate that those particles are consistent with other demographic columns. Only generate 
this rule if the sample data across the tables being validated in this pipeline run
actually contains such particles.
- Malay/Indian Singapore example: `Bin` or `s/o` indicates Male; `Binte` or `d/o` indicates
  Female — flag if `Gender` contradicts the particle.
- Spanish example: gendered suffixes in honorifics must match the `Gender` column.
---

## Pre-submission checklist
 
Before returning your output, verify each rule against these gates:
 
1. **Type is correct** — no `custom` where a named type exists (see type-selection table).
2. **Logic polarity** — every `logic` expression returns `True` for the **bad** case.
   If `not pd.isna(...)` appears, it is almost certainly inverted.
3. **Sentinel rules have values** — every `sentinel_check` rule has a non-empty
   `sentinel_values` list.
4. **Range bounds are domain-derived** — no `range` rule whose `min`/`max` is simply
   copied from the observed sample without a real domain reason.
5. **not_null rules are justified** — each one covers a mandatory field or a column with
   observed nulls (`missing_pct > 0`).
6. **Sample verification** — the rule flags observed anomalies without rejecting
   structurally valid records from the sample.
 
---

## Table context
 
### Table: {table_name}
 
### Column evidence:
{evidence_json}
 
### Sample records (first {n_sample} rows as JSON):
{sample_records_json}
 
### Cross-table join hints:
{join_hints_json}

### Dataset description:
{dataset_description}

### Other tables in this pipeline run:
{sibling_evidence_json}

Generate all rules now. Be thorough. For every column, either generate a rule or explicitly
justify why no rule is needed. If you identify a strong semantic rule but there are no
failing records in the provided sample, output the rule anyway.
"""



APPLY_VALIDATION_RULES_PROMPT = """
You are a data quality analyst applying pre-defined validation rules to dataset records.
 
Do not add, remove, rewrite, or reinterpret any rule. Apply every rule exactly as written.
If instructions below appear to conflict, lower-numbered instructions take precedence.
 
## Instructions
 
1. Evaluate **every** rule against **every** record.
2. Return **every** `rule_id` in the output, even when no rows fail.
3. Use `_row_index` values (original zero-based row index in the full dataset), not the record's
   position in the JSON array.
4. Never return a `_row_index` that is absent from the provided records.
5. Null values fail `not_null` rules. For `custom` rules, if a null value would cause
   the logic expression to error, treat the record as passing (not a violation) unless
   the rule explicitly targets null values.
 
## Type-specific behaviour
- **`enumeration`** — flag the record if the column value is not in `check_params.values`.
  Compare as strings: cast the record's column value to string before checking membership.
- **`referential_cross_table`** — compare the foreign-key value against
  `check_params.pk_values`.
- **`cross_table_semantic`** — a `sibling_lookup` dict is pre-injected into
  `check_params.sibling_lookup` as `{join_key: sibling_value}`. Look up the current
  record's `check_params.join_col` value in `sibling_lookup` to retrieve the sibling
  value, then apply `check_params.semantic_check` to determine if the record violates
  the rule. If the join key is not found in `sibling_lookup`, treat the record as passing.
- **`custom`** — evaluate the `check_params.logic` expression; the expression returns
  `True` when the row **violates** the rule.
 
## Required output
 
Return JSON only — no preamble, no explanation.
 
```json
[
  {{
    "rule_id": 1,
    "failing_record_indices": [0, 5]
  }},
  {{
    "rule_id": 2,
    "failing_record_indices": []
  }}
]
```
 
## Validation rules
 
{rules_json}
 
## Records
 
{records_json}

Before returning, verify each flagged index: confirm the record at that index actually
violates the rule as written. If uncertain, omit the index rather than include it.
"""
