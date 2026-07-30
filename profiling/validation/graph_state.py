from typing import Annotated
import operator
from typing_extensions import TypedDict

class PipelineState(TypedDict):
    validation_rules:           dict
    revision_count:             int
    regeneration_count:         int
    tables_to_regenerate:       list[str]
    validation_check_results:   Annotated[dict, operator.or_]
    inspection_notes:           list[str]
    revision_history:           Annotated[list, operator.add]

