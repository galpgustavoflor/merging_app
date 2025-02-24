from enum import Enum, auto
from typing import List

# Supported file types
FILE_TYPES: List[str] = ["csv", "xlsx"]

class Column(Enum):
    """Column names used in validation results"""
    NAME = 'Column Name'
    RULE = 'Validation Rule'
    PASS = 'Pass Count'
    FAIL = 'Fail Count'
    PERCENTAGE = 'Pass Percentage'

class ValidationRule(Enum):
    """Available validation rules"""
    VALIDATE_NULLS = 'validate_nulls'
    VALIDATE_NULLs = 'validate_nulls'
    VALIDATE_UNIQUENESS = 'validate_uniqueness'
    VALIDATE_LIST_OF_VALUES = 'validate_list_of_values'
    VALIDATE_REGEX = 'validate_regex'
    VALIDATE_RANGE = 'validate_range'
    MIN_VALUE = 'min_value'
    MAX_VALUE = 'max_value'

class Functions(Enum):
    """Available mapping functions"""
    DIRECT = 'Direct mapping'
    CONVERSION = 'Conversion mapping'
    AGGREGATION = 'Aggregation'
    #TRANSFORMATION = 'custom_transformation'
    #CONCATENATION = 'concatenation'
    #SPLIT = 'split'

class Step(Enum):
    """Enumeration of application steps"""
    SOURCE_UPLOAD = 1
    TARGET_UPLOAD = 2
    MAPPING_RULES = 3
    MATCHING = 4
    VALIDATION_RULES = 5
    DATA_VALIDATION = 6
    REPORT_SUMMARY = 7

STEP_LABELS = {
    Step.SOURCE_UPLOAD: "Upload Source",
    Step.TARGET_UPLOAD: "Upload Target",
    Step.MAPPING_RULES: "Define Mapping",
    Step.MATCHING: "Execute Matching",
    Step.VALIDATION_RULES: "Define Validation",
    Step.DATA_VALIDATION: "Execute Validation",
    Step.REPORT_SUMMARY: "View Report"
}

# Business Rule Operators
COMPARISON_OPERATORS = {
    'equals': '==',
    'not_equals': '!=',
    'greater_than': '>',
    'less_than': '<',
    'greater_equal': '>=',
    'less_equal': '<=',
    'contains': 'contains',
    'is_null': 'is null',
    'is_not_null': 'is not null'
}

LOGICAL_OPERATORS = ['AND', 'OR']

# Example business rules for guidance
EXAMPLE_RULES = [
    {
        'name': 'Active Account Balance Check',
        'conditions': [
            {'column': 'status', 'operator': 'equals', 'value': 'Active'}
        ],
        'then': [
            {'column': 'balance', 'operator': 'greater_than', 'value': '0'}
        ]
    },
    {
        'name': 'Premium Customer Discount Rule',
        'conditions': [
            {'column': 'customer_type', 'operator': 'equals', 'value': 'Premium'}
        ],
        'then': [
            {'column': 'discount', 'operator': 'less_equal', 'value': '20'}
        ]
    },
    {
        'name': 'Sales Department Region Check',
        'conditions': [
            {'column': 'department', 'operator': 'equals', 'value': 'Sales'}
        ],
        'then': [
            {'column': 'region', 'operator': 'is_not_null', 'value': ''}
        ]
    }
]
