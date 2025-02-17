from enum import Enum
from typing import List

FILE_TYPES: List[str] = ["csv", "xlsx"]

class Column:
    ORIGINAL = "Original Column"
    NEW_NAME = "New Name"
    NAME = "Column"
    NULL_VALUES = "Null Values"

class ValidationRule(Enum):
    VALIDATE_NULLS = "validate_nulls"
    VALIDATE_UNIQUENESS = "validate_uniqueness"
    VALIDATE_LIST_OF_VALUES = "validate_list_of_values"
    VALIDATE_REGEX = "validate_regex"
    VALIDATE_RANGE = "validate_range"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"

class RangeValidation:
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"

class Functions(Enum):
    DIRECT_MATCH = "Direct Match"
    AGGREGATION = "Aggregation"
    CONVERSION = "Conversion"
