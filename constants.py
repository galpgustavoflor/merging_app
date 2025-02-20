from enum import Enum
from typing import List

FILE_TYPES: List[str] = ["csv", "xlsx"]

class Column(Enum):
    NAME = 'name'
    AMOUNT = 'amount'
    DATE = 'date'
    ID = 'id'
    VALUE = 'value'
    MATCHED = 'matched'
    NULL_VALUES = ['', 'null', 'NULL', 'nan', 'NaN', 'None']

class ValidationRule(Enum):
    VALIDATE_NULLS = "validate_nulls"
    VALIDATE_NULLs = "validate_nulls"
    VALIDATE_UNIQUENESS = "validate_uniqueness"
    VALIDATE_LIST_OF_VALUES = "validate_list_of_values"
    VALIDATE_REGEX = "validate_regex"
    VALIDATE_RANGE = "validate_range"
    MIN_VALUE = "min_value"
    MAX_VALUE = "max_value"

class Functions(Enum):
    DIRECT_MATCH = "Direct Match"
    AGGREGATION = "Aggregation"
    CONVERSION = "Conversion"

class Step:
    SOURCE_UPLOAD = 1
    TARGET_UPLOAD = 2
    MAPPING_RULES = 3
    MATCHING = 4
    VALIDATION_RULES = 5
    DATA_VALIDATION = 6
    REPORT_SUMMARY = 7  # New final step for report summary

STEP_LABELS = {
    Step.SOURCE_UPLOAD: "Source Upload",
    Step.TARGET_UPLOAD: "Target Upload",
    Step.MAPPING_RULES: "Mapping Rules",
    Step.MATCHING: "Matching",
    Step.VALIDATION_RULES: "Validation Rules",
    Step.DATA_VALIDATION: "Data Validation",
    Step.REPORT_SUMMARY: "Report Summary"  # New label
}
