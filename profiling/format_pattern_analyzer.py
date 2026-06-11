"""
Format Pattern Analyzer - Detects data format patterns for better LLM prompts

This runs BEFORE LLM generation to provide evidence-based context.
"""

import re
import pandas as pd
from collections import Counter
from typing import Dict
from .config import PLACEHOLDER_TOKENS

class FormatPatternAnalyzer:
    """
    Analyzes column values to detect format patterns and anomalies.
    
    Provides concrete evidence for LLM to generate specific recommendations.
    """
    
    def __init__(self,config=None):
        # Common format patterns
        self.config = config
        self.patterns = {
            #'phone_sg': r'^[689]\d{7}$',
            'phone_intl': r'^\+\d{1,3}[-.\s]?\d{1,14}$',
            'email': r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$',
            #'nric_sg': r'^[STFGM]\d{7}[A-Z]$',
            'date_iso': r'^\d{4}-\d{2}-\d{2}$',
            #'date_sg': r'^(0?[1-9]|[12]\d|3[01])/(0?[1-9]|1[0-2])/\d{4}$',
            #'postal_code_sg': r'^\d{6}$',
            'currency': r'^\$\d{1,3}(,\d{3})*(\.\d{2})?$',
            'url': r'^https?://[^\s]+$',
            'uuid': r'^[{]?[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}[}]?$',
        }
        if getattr(self.config, "enable_country_specific_patterns", False):
            self.patterns.update({
                "phone_sg": r"^[689]\d{7}$",
                "nric_sg": r"^[STFGM]\d{7}[A-Z]$",
                "postal_code_sg": r"^\d{6}$",
            })
    
    def analyze_format_distribution(
        self, 
        series: pd.Series,
        sample_size: int = 1000
    ) -> Dict:
        """
        Analyze format patterns in a column.
        
        Returns detailed breakdown of formats, patterns, and anomalies.
        """
        # Sample for efficiency
        non_null = series.dropna()
        if len(non_null) == 0:
            return {'status': 'empty'}
        
        sample = non_null.sample(min(sample_size, len(non_null)), random_state=42)
        sample_str = sample.astype(str)
        
        # 1. Detect known patterns
        pattern_matches = {}
        for pattern_name, pattern_regex in self.patterns.items():
            matches = sample_str.str.match(pattern_regex, case=False)
            match_count = matches.sum()
            if match_count > 0:
                pattern_matches[pattern_name] = {
                    'count': int(match_count),
                    'percentage': round(match_count / len(sample) * 100, 1),
                    'examples': sample_str[matches].head(3).tolist()
                }
        
        # 2. Abstract format fingerprints
        format_fingerprints = self._extract_format_fingerprints(sample_str)
        
        # 3. Identify anomalies
        anomalies = self._find_anomalies(non_null,sample_str, format_fingerprints)
        
        # 4. Check if formats are coercible
        coercibility = self._assess_coercibility(format_fingerprints)
        
        # 5. Calculate uniformity score
        uniformity = self._calculate_uniformity(format_fingerprints)
        
        return {
            'total_values': len(non_null),
            'sample_size': len(sample),
            'known_patterns': pattern_matches,
            'format_fingerprints': format_fingerprints,
            'anomalies': anomalies,
            'coercibility': coercibility,
            'uniformity_score': uniformity,
            #'recommendation_tier': self._recommend_tier(uniformity, coercibility, anomalies)
        }
    
    def _extract_format_fingerprints(self, series: pd.Series) -> Dict:
        """
        Convert values to abstract format patterns.
        
        Example:
            "(123) 456-7890" → "(XXX) XXX-XXXX"
            "555-1234" → "XXX-XXXX"
            "abc-123" → "aaa-XXX"
        """
        def to_fingerprint(val):
            val = str(val)
            fingerprint = ""
            for char in val:
                if char.isdigit():
                    fingerprint += 'X'
                elif char.isalpha():
                    fingerprint += 'a'
                else:
                    fingerprint += char
            return fingerprint
        
        fingerprints = series.apply(to_fingerprint)
        fingerprint_counts = Counter(fingerprints)
        
        # Top 10 most common formats
        top_formats = []
        for fingerprint, count in fingerprint_counts.most_common(10):
            # Get examples
            examples = series[fingerprints == fingerprint].head(3).tolist()
            top_formats.append({
                'pattern': fingerprint,
                'count': count,
                'percentage': round(count / len(series) * 100, 1),
                'examples': examples
            })
        
        return {
            'total_unique_formats': len(fingerprint_counts),
            'top_formats': top_formats,
            'dominant_format': top_formats[0] if top_formats else None
        }
    
    def _find_anomalies(self, series: pd.Series,series_str: pd.Series, format_info: Dict) -> Dict:
        """
        Find values that don't match dominant patterns.
        
        Anomalies are values that:
        - Match rare formats (< 5% of data)
        - Contain placeholder values (N/A, UNKNOWN, etc.)
        - Have suspicious patterns
        """
        anomalies = {
            'placeholder_values': [],
            'suspicious_values': [],
        }
        
        for val in series_str:
            val_lower = str(val).lower().strip()
            if val_lower in PLACEHOLDER_TOKENS:
                anomalies['placeholder_values'].append(val)

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric) > 0:
                skewness = numeric.skew()
                if abs(skewness) > 1 or len(numeric) < 100:
                    p99 = numeric.quantile(0.999)
                    p01 = numeric.quantile(0.001)
                    iqr = numeric.quantile(0.75) - numeric.quantile(0.25)
                    upper = p99 + self.config.outlier_tail_multiplier * iqr
                    lower = p01 - self.config.outlier_tail_multiplier * iqr
                    outliers = series[
                        pd.to_numeric(series, errors='coerce').gt(upper) |
                        pd.to_numeric(series, errors='coerce').lt(lower)
                    ]
                else:
                    mean = numeric.mean()
                    std = numeric.std()
                    if std > 0:
                        z_scores = (pd.to_numeric(series, errors='coerce') - mean) / std
                        outliers = series[z_scores.abs() > self.config.outlier_z_score_threshold]
                    else:
                        outliers = series.iloc[0:0]
                anomalies['suspicious_values'] = outliers.unique().tolist()[:10]
        else:
            for val in series_str.head(100):
                val_str = str(val).lower()
                if any(re.match(p, val_str) for p in self.config.suspicious_string_patterns):
                    anomalies['suspicious_values'].append(val)

        for key in ['placeholder_values', 'suspicious_values']:
            anomalies[key] = list(set(str(v) for v in anomalies[key]))[:10]

        anomalies['total_anomaly_count'] = sum(len(anomalies[k]) for k in ['placeholder_values', 'suspicious_values'])

        if pd.api.types.is_numeric_dtype(series):
            numeric_full = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric_full) > 0:
                anomalies['col_min'] = float(numeric_full.min())
                anomalies['col_max'] = float(numeric_full.max())

        return anomalies
    
    def _assess_coercibility(self, format_info: Dict) -> Dict:
        """
        Determine if formats can be automatically standardized.
        
        Coercible: Multiple formats that represent the same semantic content
        Example: (123) 456-7890, 123-456-7890, 1234567890 → all coercible to one format
        """
        top_formats = format_info['top_formats']
        
        if not top_formats or len(top_formats) < 2:
            return {
                'is_coercible': False,
                'reason': 'Single format already' if top_formats else 'No data'
            }
        
        dominant = top_formats[0]
        dominant_pattern = dominant['pattern']

        # If all formats contain only digits and dots (e.g. X.X vs X.XX),
        # it's floating-point precision variation — not a real format issue.
        all_numeric_precision = all(
            re.sub(r'[X.]', '', f['pattern']) == ''
            for f in top_formats
        )
        if all_numeric_precision:
            return {
                'is_coercible': False,
                'reason': 'Numeric precision variation only — not a format issue'
            }
        
        all_integer_length = all(
            re.sub(r'X', '', f['pattern']) == ''
            for f in top_formats
        )
        if all_integer_length:
            return {
                'is_coercible': False,
                'reason': 'Integer length variation only — not a format issue'
            }
        all_text_structure_variation = all(
            re.sub(r"[a\s.'\-\/]", "", f["pattern"]) == ""
            for f in top_formats
        )

        if all_text_structure_variation:
            return {
                "is_coercible": False,
                "reason": "Text length/word-count variation only — not a format issue"
            }
        
        coercible_formats = []
        non_coercible_formats = []
        
        for fmt in top_formats[1:]:
            dom_stripped = re.sub(r'[^Xa]', '', dominant_pattern)
            fmt_stripped = re.sub(r'[^Xa]', '', fmt['pattern'])

            # Pure integer length variation — skip entirely, not a structural difference
            if (re.sub(r'X', '', dom_stripped) == '' and 
                    re.sub(r'X', '', fmt_stripped) == ''):
                continue

            if dom_stripped == fmt_stripped:
                coercible_formats.append(fmt)
            else:
                non_coercible_formats.append(fmt)
        
        coercible_pct = sum(f['percentage'] for f in coercible_formats)
        if not coercible_formats and dominant['percentage'] < 50:
            return {
                'is_coercible': False,
                #'reason': 'No dominant format and no coercible variants — natural variation'
            }
        
        return {
            'is_coercible': len(coercible_formats) > 0,
            'coercible_percentage': round(coercible_pct, 1),
            'coercible_formats': coercible_formats,
            'non_coercible_formats': non_coercible_formats,
            'target_format': dominant_pattern,
            'recommendation': self._coercion_recommendation(
                dominant, coercible_formats, non_coercible_formats
            )
        }
    
    def _coercion_recommendation(self, dominant, coercible, non_coercible):
        """Generate specific coercion recommendation."""
        if not coercible and not non_coercible:
            return "No action needed - format already uniform"
        
        rec = []
        
        if coercible:
            total_pct = sum(f['percentage'] for f in coercible)
            rec.append(
                f"Standardize {total_pct:.1f}% of values to {dominant['pattern']} format (coercible)"
            )
        
        if non_coercible:
            total_pct = sum(f['percentage'] for f in non_coercible)
            rec.append(
                f"Clarify with data owner: {total_pct:.1f}% use incompatible format {non_coercible[0]['pattern']}"
            )
        
        return " | ".join(rec)
    
    def _calculate_uniformity(self, format_info: Dict) -> float:
        """
        Calculate format uniformity score (0-1).
        
        1.0 = All values use same format
        0.0 = Every value uses different format
        """
        if not format_info['top_formats']:
            return 0.0
        
        dominant_pct = format_info['top_formats'][0]['percentage']
        return round(dominant_pct / 100, 3)
    
    
    def generate_llm_context(self, column_name: str, series: pd.Series, analysis: dict | None = None) -> str:
        """
        Generate rich context for LLM prompt.
        
        This replaces generic column info with specific format analysis.
        Pass precomputed analysis to avoid running it twice.
        """

        if analysis is None:
            analysis = self.analyze_format_distribution(series)
        
        context_parts = [
            f"Column: {column_name}",
            f"Total values: {analysis['total_values']:,}",
        ]
        
        # Format distribution
        top_formats = analysis.get('format_fingerprints', {}).get('top_formats', [])
        
        def is_length_variation_only(pattern: str) -> bool:
            """
            True if pattern varies only in length — not a real format issue.
            Strips spaces, dots, apostrophes, hyphens — all common in names/suburbs.
            """
            stripped = re.sub(r"[ .\'\-\/@]", '', pattern)
            if not stripped:
                return False
            return (
                all(c == 'a' for c in stripped) or
                all(c == 'X' for c in stripped)
            )

        '''all_length_variation = (
            bool(top_formats) and
            all(is_length_variation_only(f['pattern']) for f in top_formats[:5])
        )'''

        meaningful_formats = [
            fmt for fmt in top_formats[:5]
            if not is_length_variation_only(fmt['pattern'])
        ]

        # Coercibility — compute FIRST so all_flagged_length_variation is available
        coercion = analysis.get('coercibility', {})
        recommendation = coercion.get('recommendation', '')
        non_coercible = coercion.get('non_coercible_formats', [])
        coercible_fmts = coercion.get('coercible_formats', [])

        all_flagged_length_variation = all(
            is_length_variation_only(f.get('pattern', ''))
            for f in (non_coercible + coercible_fmts)
        ) if (non_coercible or coercible_fmts) else True

        has_real_format_issue = (
            coercion.get('is_coercible') or bool(non_coercible)
        ) and not all_flagged_length_variation

        known_patterns = analysis.get("known_patterns", {})
        total_known_pct = sum(info["percentage"] for info in known_patterns.values())
        if total_known_pct >= 80:
            has_real_format_issue = False

        if has_real_format_issue and meaningful_formats:
            context_parts.append("\nFormat Distribution:")
            for fmt in meaningful_formats[:3]:
                context_parts.append(
                    f"  - {fmt['percentage']:.1f}%: {fmt['pattern']} (e.g., {fmt['examples'][0]})"
                )
            uniformity = analysis.get('uniformity_score', 0)
            context_parts.append(f"\nFormat Uniformity: {uniformity:.1%}")

        # Known patterns
        if known_patterns:
            context_parts.append("\nDetected Patterns:")
            for pattern_name, info in known_patterns.items():
                context_parts.append(
                    f"  - {pattern_name}: {info['percentage']:.1f}%"
                )

        # Anomalies
        anomalies = analysis.get('anomalies', {})
        if anomalies.get('total_anomaly_count', 0) > 0:
            context_parts.append(f"\nAnomalies Found: {anomalies['total_anomaly_count']}")
            if anomalies.get('placeholder_values'):
                context_parts.append(
                    f"  - Placeholder values: {anomalies['placeholder_values'][:3]}"
                )

        # Coercibility recommendation — reuse already-computed variables
        if (has_real_format_issue
                and recommendation
                and not all_flagged_length_variation
                and recommendation != "No action needed - format already uniform"):
            context_parts.append(f"\nFormat Analysis: {recommendation}")


        return "\n".join(context_parts)

