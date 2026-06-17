"""
Shared constants and default configuration for the profiling pipeline.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

PLACEHOLDER_TOKENS: set[str] = {
        'n/a', 'na', 'null', 'none', 'unknown', 'tbd', 'pending',
        'n.a.', 'not available', '--', '???', '?', 'missing'
    }

#EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+$")

ID_NAME_HINTS: set[str] = {"id", "key", "identifier", "uuid", "ssn", "soc_sec"}

SUPPORTED_EXTENSIONS: list[str] = [".csv", ".xlsx", ".xls", ".parquet", ".json",".geojson", ".txt"]


@dataclass
class PipelineConfig:
    """
    Central configuration object.  Pass one instance through the pipeline
    so every component reads from the same source of truth.
    """

    # Directories
    data_dir: Path = Path("data/raw")
    output_dir: Path = Path("profile_outputs")

    # Preprocessing
    categorical_threshold: int = 20

    # Column analysis
    max_sample_values: int = 5
    max_permissible_values: int = 50
    near_dupe_threshold: float = 0.88
    near_dupe_num_perm: int = 64
    near_dupe_k: int = 3
    near_dupe_min_values: int = 2
    near_dupe_max_values: int = 2000

    # MinHash / LSH
    minhash_num_perm: int = 128
    minhash_shingle_k: int = 3
    join_threshold: float = 0.5
    shingle_join_threshold: float = 0.6
    duplicate_threshold: float = 0.9
    near_identical_threshold: float = 0.7

    # Composite quality score approach
    join_quality_threshold: float = 0.65    
    min_join_cardinality: int = 5      
    min_overlap_count: int = 500  
    enable_country_specific_patterns: bool = False

    # Relationship detection
    coverage_join_threshold: float = 0.90
    pk_uniqueness_threshold: float = 0.95
    fk_coverage_threshold: float = 0.95
    one_to_one_coverage_threshold: float = 0.95

    # LLM
    llm_chunk_size: int = 1
    llm_max_retries: int = 3
    llm_save_raw_outputs: bool = True
    llm_validation_sample_size: int = 50
    llm_resume: bool = False
    llm_timeout: int = 600
    llm_model: str = "" 
    llm_chunk_model: str = ""         # Azure deployment name for JSON chunks, e.g. "gpt-4o-mini"
    llm_endpoint: str = ""            # Azure endpoint for primary model
    llm_chunk_endpoint: str = ""      # Azure endpoint for chunk model (falls back to llm_endpoint)
    llm_is_native_azure: bool = False  # True = AzureOpenAI client, False = OpenAI-compatible client
    llm_chunk_is_native_azure: bool = False

    
    outlier_z_score_threshold: float = 3.0
    outlier_tail_multiplier: float = 2.0
    suspicious_string_patterns: list[str] = field(default_factory=lambda: [
        r'^0{3,}$',
        r'^1{3,}$',
        r'^9{3,}$',
        #r'^test',
        r'^x{3,}',
        r'^dummy',
    ])

    # Cross-column date ordering — semantic name hints
    date_ordering_start_hints: set[str] = field(default_factory=lambda: {
        "start", "begin", "open", "created", "sell"
    })
    date_ordering_end_hints: set[str] = field(default_factory=lambda: {
        "end", "close", "expir", "discontinu"
    })
    date_ordering_update_hints: set[str] = field(default_factory=lambda: {
        "modified", "updated", "changed"
    })

        # Relationship-name heuristics
    # These are generic defaults, not dataset-specific rules.
    # They can be tuned per project if needed.
    relationship_descriptive_prefixes: set[str] = field(default_factory=lambda: {
        "billing", "shipping", "mailing",
        "home", "work", "customer", "supplier", "vendor",
        "contact", "delivery", "invoice"
    })

    relationship_descriptive_terms: set[str] = field(default_factory=lambda: {
        "name", "firstname", "lastname", "fullname",
        "address", "street", "city", "state", "province", "region",
        "country", "postalcode", "postcode", "zipcode", "zip",
        "phone", "mobile", "fax", "email",
        "title", "description", "comment", "notes",
    })

    relationship_hierarchy_terms: set[str] = field(default_factory=lambda: {
        "reportsto", "manager", "managerid",
        "parent", "parentid",
        "supervisor", "supervisorid",
        "boss", "lead", "owner", "ownerid"
    })

    relationship_assignment_parent_terms: set[str] = field(default_factory=lambda: {
        "employee", "staff", "person", "people",
        "user", "agent", "representative", "rep",
        "accountmanager"
    })

    relationship_assignment_fk_terms: set[str] = field(default_factory=lambda: {
        "rep", "representative", "agent", "staff",
        "employee", "salesperson", "owner",
        "manager", "assignee"
    })

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)