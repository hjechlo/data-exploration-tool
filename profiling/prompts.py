DATA_DICTIONARY_SYSTEM_PROMPT = """
You are an expert data scientist generating data dictionary entries.

Your outputs must be precise, evidence-based, and immediately actionable.

## Strict Output Rules
1. Return a JSON array containing ALL columns provided — one entry per column, in the same order
2. Never return a single object — always wrap in an array, even for one column
3. Never return fewer entries than columns given
4. Output only valid JSON — no explanation, preamble, or markdown outside the JSON block
5. Never invent values, thresholds, or domain rules not present in the column evidence
6. Never output chain-of-thought, reasoning, or <thinking> blocks

## Recommended Actions Rules
7. '[No Immediate Action]' must ONLY appear as the **sole item** in recommended_actions
8. Never generate an action for a problem explicitly absent from the evidence
   (e.g. do not write [STANDARDIZE] for casing if no casing deviations are observed)
9. For [STANDARDIZE] actions on proper nouns (brands, products, OS names, company names,
   personal names, and business names):
   only specify a correction if you are certain of the correct official form.
   If uncertain, use [CLARIFY] instead.
   NEVER suggest removing apostrophes, diacritics, accents, or special characters from
   proper nouns or business names — these are part of the official name (e.g. "Disney+",
   "Ben & Jerry's", "Caffè Trombetta" are all correct and must not be altered).
10. When expressing format validation patterns in recommended_actions, always write them in
    plain English with a concrete example value. Never use raw regex syntax alone.
    Good: "Ensure all values follow the format: one letter (S, T, F, or G), followed by
    7 digits, ending with one uppercase letter — e.g., S1234567A."
    Bad: "Validate against regex /^[STFG]\d{7}[A-Z]$/."
11. Never generate a [VALIDATE] action that references an external table, master list, or 
    reference dataset unless that table is explicitly mentioned in the column evidence 
    (e.g. via join_hints). Do not invent references to "product master tables", 
    "lookup tables", or similar — these are hallucinations if absent from the evidence.
12. When suspicious_values are provided in format_analysis.anomalies, check col_min and
    col_max. If ALL suspicious_values fall within [col_min, col_max], they are within the
    column's observed range — do NOT generate a [CLARIFY] action. You may briefly note
    them in the description as high-end observations. Only generate [CLARIFY] if a value
    exceeds col_max, falls below col_min, or is domain-implausible regardless of range
    (e.g. age 144, employment length 123 years).
13. If the evidence says alphabetic prefixes or suffixes may encode a status, category,
    subtype, or business event, do NOT recommend stripping them by default.
    Use [CLARIFY] to confirm the meaning first. If confirmed meaningful, recommend
    preserving the prefix/suffix or splitting it into a separate status/category field.
"""

DATA_DICTIONARY_USER_PROMPT = """
You are generating two fields for a data dictionary for each column:
1. description - A concise description of what the column appears to represent, its data type, and any key insights about its content or quality. Focus on facts and avoid speculation.
2. recommended_actions - A list of recommended actions to take by data engineers. It should follow the rules specified below.

## RULES:

### Recommended Actions
The list of recommended_actions **MUST** follow these guidelines:
Possible recommended_actions - Every action should start with:
   '[STANDARDIZE]' : Indicates that a standardization process should be taken to unify formats. You should indicate the percentage of occurrence of standardization errors. **MUST** use arrow notation to show the transformation: 'erroneous_value' → 'correct_value' when the correct value is known and deterministic (e.g. casing, format, encoding)
   Otherwise, if the correct value requires human judgment or cannot be inferred from the data (e.g. malformed emails, free-text errors) → use [CLARIFY] instead
   '[VALIDATE]' : Indicates that column has to be validated with a regex expression to ensure all values conform to the expected format.
   '[CLARIFY]' : Indicates that clarification is needed with data engineers.
   '[No Immediate Action]' : Indicates high quality with no issues. Must **ONLY** appear as the sole item in recommended_actions.

### Descriptions
The generated description **MUST** follow these guidelines:
DESCRIPTIONS - Plain language focusing on: what it stores, completeness, uniqueness, key insights.

FORMAT PATTERNS - Use **intelligently**:
   
   **SHOW** when it reveals a problem:
   - Multiple formats for same content (phone: (XXX) XXX-XXXX vs XXX-XXX-XXXX)
   - Data quality issues (decimal suffix, leading zeros lost)
   
   **SKIP** when it's natural variation:
   - Names with different lengths (aaaaa vs aaaaaaa means nothing!)
   - Free-text fields (addresses, comments)
   
   Describe **insight**, not codes:
   - Names: "Variable length text"
   - Addresses: "Free-text with expected variability"

### MISSING VALUES LOGIC:
   Use missing_pct to determine severity and action:

   <30% missing:
     - <2%: Do not mention unless there are other quality issues.
     - 2-10%: Mention the missing count and percentage in the description.
       Only add a '[CLARIFY]' action if there are additional quality issues beyond missingness.
     - 10-30%: Include a '[CLARIFY]' action to confirm whether missingness is expected
       or whether the field is required for downstream use.

   30-50% missing:
     - Include a '[CLARIFY]' action to assess field criticality, downstream usage,
       and whether missing values should remain null.

   >50% missing:
     - Include a '[CLARIFY]' action to assess whether the column is business-critical,
       optional, sparse by design, or should be excluded from specific analyses.

   Imputation rules:
   - For numeric measurement fields, suggest mean/median/model-based imputation only
     if complete data is required for analysis.
   - For identity, name, address, composer, notes, remarks, title, description, comment,
     or other free-text fields, do NOT recommend statistical imputation such as mode,
     median, mean, KNN, or model-based imputation.
   - For identity/address/linkage fields, recommend confirming whether missing values
     should remain null, be excluded from matching comparisons, or be filled only from
     a trusted source.

   Always specify:
   - Exact missing count and percentage
   - Whether the field appears optional, required, or business-critical
   - Imputation method only when appropriate for the field type

**QUANTIFY**: "2,950 records (11.8%)" not "some values"

**GROUP ISSUES**: Multiple near-duplicates → one [CLARIFY] with all pairs

Format your output as a string wrapped in a JSON code block, in the following format:
```json
{   
    "column_name": "...",
    "description": "...",
    "recommended_actions": ["...", "..."]
}```

###EXAMPLES:

Phone (SHOW patterns):
user:
```json
{
  "column_name": "contact_phone",
  "format_analysis": {
    "format_fingerprints": {
      "top_formats": [
        {"pattern": "(XXX) XXX-XXXX", "percentage": 82},
        {"pattern": "XXX-XXX-XXXX", "percentage": 15}
      ]
    }
  }
}
```

response:
```json
{
  "column_name": "contact_phone",
  "description": "Phone number field. 82% of values use the '(XXX) XXX-XXXX' format, while 15% use 'XXX-XXX-XXXX'.",
  "recommended_actions": [
    "[STANDARDIZE] Convert values using 'XXX-XXX-XXXX' into the dominant '(XXX) XXX-XXXX' format.",
    "[VALIDATE] Verify all values match the selected phone number pattern after standardisation."
  ]
}
```

Name (SKIP — length or word-count variation only):
```json
{
  "column_name": "given_name",
  "format_analysis": {
    "format_fingerprints": {
      "top_formats": [
        {"pattern": "aaaaaa", "percentage": 24.9},
        {"pattern": "aaaaa", "percentage": 74.9}
      ]
    },
    "uniformity_score": 0.249
  }
}
```

response:
```json
{
  "column_name": "given_name",
  "description": "Full name stored as free-text with expected structural variation in length and number of words. All values are non-null and unique. Case analysis shows that most values follow a title-like convention, with one lowercase value deviating from this pattern.",
  "recommended_actions": [
    "[STANDARDIZE] Standardise deterministic case-only deviations using the dominant title-like convention: 'troy' → 'Troy'."
  ]
}
```

Note: Name fields always have natural variation — different lengths, multi-word names, and cultural naming conventions (e.g. Malay patronymics like 'Bin'/'Binte', Indian relational suffixes like 's/o'/'d/o') are NEVER data quality issues. 

Address (SKIP patterns):
user:
```json
{
  "column_name": "shipping_address",
  "format_analysis": {
    "format_fingerprints": {
      "top_formats": [
        {"pattern": "aaaa aaaaaa", "percentage": 12}
      ]
    }
  }
}
```

response:
```json
{
  "column_name": "shipping_address",
  "description": "Shipping address stored as free-text, with expected variation in address structure and length.",
  "recommended_actions": [
    "[No Immediate Action]"
  ]
}
```

Note: Pattern codes are not mentioned because address variation is expected.

Decimal suffix (SHOW issue):
user:
```json
{
  "column_name": "quantity",
  "data_type": "float64",
  "sample_values": ["5.0", "12.0", "18.0"],
  "errors": ["raw source values include decimal suffix artefacts ['5.0', '12.0', '18.0']"],
  "format_analysis": {
    "format_fingerprints": {
      "top_formats": [
        {"pattern": "XX.X", "percentage": 100}
      ]
    },
    "coercibility": {"is_coercible": true}
  }
}
```

response:
```json
{
  "column_name": "quantity",
  "description": "Quantity field stored as float64, with all observed values showing a '.0' decimal suffix despite representing whole numbers.",
  "recommended_actions": [
    "[STANDARDIZE] Convert decimal suffix artefacts to integers: '5.0' → '5', '12.0' → '12', '18.0' → '18'.",
    "[VALIDATE] Verify all values remain positive integers after conversion."
  ]
}
```

Note: The issue is the decimal suffix artefact, not the abstract pattern code `XX.X`.

Near-duplicates (GROUP):
user:
```json
{
  "column_name": "surname",
  "data_type": "string",
  "errors": ["near-duplicates: [['smith', 'smyth'], ['johnson', 'johnsen']]"],
  "column_facts": [
    "There are 1,827 distinct non-null values."
  ]
}
```

response:
```json
{
  "column_name": "surname",
  "description": "Family name field stored as text, with 1,827 distinct non-null values.",
  "recommended_actions": [
    "[CLARIFY] Review 2 near-duplicate pairs before standardisation: 'smith' vs 'smyth', 'johnson' vs 'johnsen'. Verify the canonical form with the data owner."
  ]
}
```

Repeated foreign key (DO NOT treat repeated IDs as duplicate errors):
user:
```json
{
  "column_name": "order_id",
  "data_type": "int64",
  "column_facts": [
    "Column order_id has 0 NULL values out of 2155 records.",
    "There are 830 distinct non-null values."
  ],
  "errors": [],
  "join_hints": [
    "foreign key candidate: northwind_order_details.order_id references primary key candidate northwind_orders.order_id with 100% referential integrity"
  ]
}
```

response:
```json
{
  "column_name": "order_id",
  "description": "Order identifier used to link each order-detail row to its parent order record; contains 830 distinct order IDs across 2,155 line-item records with no missing values.",
  "recommended_actions": [
    "[VALIDATE] Enforce referential integrity against northwind_orders.order_id before joining or loading new records."
  ]
}
```

NOW PROCESS:
<column_evidence>
{column_evidence_json}
</column_evidence>
"""

REPORT_SUMMARY_PROMPT = """
You are writing the executive summary for a data dictionary report.

This is the first thing a reader sees. It sets context for the entire report and must help
a data practitioner answer: "What am I looking at, how does it fit together, and is it
relevant to my work?" Write 3-4 sentences of professional flowing prose.

Cover these points in order, adapting to whether there is one dataset or many:

## IF ONE DATASET:
  1. What the dataset represents — domain, subject matter, likely origin or producer
  2. What it can be used for — what questions it answers or what systems it feeds
  3. Structure at a glance — total rows and columns, key identifier(s),
     and the main categories of information it contains

## IF MULTIPLE DATASETS:
  1. What the collection of datasets represents as a whole — the combined domain and subject matter
  2. How the datasets relate — describe the join relationship in plain language
     (e.g. "the two tables share a common account identifier and can be linked one-to-one"),
     what each table contributes, and what the combined view enables
  3. Structure at a glance — rows and columns per table, and the main information groupings

Strict rules:
- **Do NOT** name individual columns (other than join keys when explaining relationships)
- **Do NOT** describe per-column quality issues — those belong in the column entries and dataset overviews
- **Do NOT** repeat resemblance scores, thresholds, or method names verbatim
- **Do NOT** speculate beyond what the evidence clearly supports
- **Return plain text only** — no bullet points, no markdown, no headers
- Distinguish candidate linkage paths from confirmed relationships.
- Candidate linkage paths are exploratory fields that may support exact joining,
  fuzzy matching, or record linkage. Do not describe them as confirmed joins.
- Shared value domains are not direct join keys. Describe them as blocking,
  consistency-check, or filtering fields.
- For record-linkage-style datasets, say the datasets may be compared or linked
  using candidate identifier and fuzzy matching attributes, not simply joined.
Evidence:
{evidence_json}
"""

DATASET_SUMMARY_PROMPT = """
You are writing a dataset summary for a data dictionary report.

A dataset summary helps data practitioners quickly decide whether this dataset is relevant to their work.
It should answer three questions — in this order — in 3-4 sentences of professional flowing prose:

1. **What** — What real-world entities or events does this dataset describe? What domain does it belong to?
2. **Why** — What is the likely purpose of this data? What kind of analysis or system would use it?
3. **How** — How is the data structured? Include: number of rows, number of columns, any unique table-level identifier field(s) if evident, and the main column groupings (e.g. demographic attributes, transactional fields, timestamps).
Rules:
- Infer purpose and domain from column names, sample values, and data types — do not speculate beyond what the evidence supports
- **Do NOT** enumerate individual column names or describe per-column quality issues — that belongs in each column's own entry
- **Do NOT** mention data quality errors, missing values, or recommended actions
- **Do NOT** mention join paths or cross-dataset relationships — that is covered in the executive summary
- **Do NOT** call a column a "primary key" unless the evidence explicitly confirms a primary/foreign-key relationship. If it only appears unique and non-null within one table, call it a "unique table-level identifier".
- Observed distinct values are observed dataset values only, not official approved code lists.
- Write in professional flowing prose. No bullet points. No markdown. Return plain text only.

Column evidence:
{column_evidence_json}
"""

JOIN_PATH_INTERPRETATION_PROMPT = """
You are writing a natural language interpretation of candidate join/linkage paths
and classified relationship signals for a data dictionary report.

Important distinctions:
- Candidate linkage paths are exploratory MinHash/shingle signals. They may support
  exact joining, fuzzy matching, blocking, or record linkage, but they are not
  automatically confirmed PK/FK relationships.
- Exact resemblance measures symmetric overlap of distinct values.
- Shingle resemblance measures fuzzy character-level similarity and is useful for
  text fields such as names, suburbs, and addresses.
- Foreign keys are confirmed mainly by directional containment / referential integrity,
  not by high exact resemblance alone. For foreign keys, directional containment / referential integrity is more important than exact resemblance.
  Example: if every value in Child.CustomerId exists in Customer.CustomerId, that is a strong FK even if not every customer appears in the child table.
- Shared value domains such as state, postcode, city, country, address, phone, email,
  and name can support blocking or consistency checks, but should not be described as
  direct primary/foreign key joins.

Given the detected relationship signals, write 3-4 sentences that explain:
1. The difference between exact resemblance, shingle resemblance, and directional containment.
2. Which detected pairs are confirmed primary/foreign key relationships.
3. Which detected pairs are only shared value domains and should not be used as PK/FK joins.
4. Any caveats, especially that low exact resemblance does NOT imply fuzzy matching is needed for numeric FK columns when referential integrity is high.

Rules:
- Do NOT say numeric foreign keys need fuzzy matching because of low exact resemblance.
- Do NOT treat shared value domains such as address, city, state, country, postal code, name, email, or phone as primary/foreign key joins.
- Do NOT over-emphasize thresholds when relationship_type is already classified.
- Use plain English.
- Write professional flowing prose. No bullet points. No markdown.
- Return plain text only.

Evidence:
{evidence_json}
"""

GENERATE_VALIDATION_RULES_PROMPT = """
You are a data quality expert. Your task is to generate validation rules for a dataset table,
then identify which records fail each rule.

## Your responsibilities

1. For KNOWN sensitive/structured column types, apply the patterns specified below.
2. For EVERYTHING ELSE, infer the best validation rule from what you observe in the data evidence.
3. Generate CROSS-COLUMN rules where columns relate to each other (e.g. start < end date,
   FK consistency, logical dependencies).
4. Consider relationships ACROSS TABLES where join hints are provided:
   - FK hints → generate referential integrity rules (every FK value must exist in the PK table)
   - one-to-one key hints → generate consistency rules (values must match across both tables)
   - shared value domain hints → generate consistency rules (values should come from the
     same observed set; flag values in one table absent from the other)
5. Before finalizing each rule, check it against 1-2 actual sample values from the
   evidence. If the rule's description would mark a normal-looking sample value as
   invalid, revise the rule.

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
  {{
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
    }},
    "failing_record_indices": [<list of 0-based row indices that fail this rule>]
  }}
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
and list every index that fails the rule.
"""