from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class ValidationRule:
    validate_nulls: bool = False
    validate_uniqueness: bool = False
    validate_list_of_values: Optional[List[str]] = None
    validate_regex: Optional[str] = None
    validate_range: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None

@dataclass
class MappingRule:
    destinations: List[str]
    function: str
    transformation: Any

@dataclass
class MappingConfig:
    key_source: List[str]
    key_target: List[str]
    mappings: Dict[str, MappingRule]
