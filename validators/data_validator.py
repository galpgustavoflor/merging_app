from typing import Dict, Any, List
import pandas as pd
from .base import BaseValidator
from constants import ValidationRule as VRule

class DataValidator(BaseValidator):
    def __init__(self, validation_rules: Dict[str, Any]):
        self.validation_rules = validation_rules

    def validate(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        results = []
        for column, rules in self.validation_rules.items():
            if not column in data.columns:
                continue
                
            column_results = []
            if rules.get(VRule.VALIDATE_NULLS.value):
                column_results.append(self._validate_nulls(data, column))
            if rules.get(VRule.VALIDATE_UNIQUENESS.value):
                column_results.append(self._validate_uniqueness(data, column))
            if rules.get(VRule.VALIDATE_LIST_OF_VALUES.value):
                column_results.append(self._validate_allowed_values(data, column, rules[VRule.VALIDATE_LIST_OF_VALUES.value]))
            if rules.get(VRule.VALIDATE_REGEX.value):
                column_results.append(self._validate_regex(data, column, rules[VRule.VALIDATE_REGEX.value]))
            if rules.get(VRule.VALIDATE_RANGE.value):
                column_results.append(self._validate_range(data, column, rules))
                
            results.extend(column_results)
        
        return results

    def get_validation_rules(self) -> Dict[str, Any]:
        return self.validation_rules

    def _validate_nulls(self, data: pd.DataFrame, column: str) -> Dict[str, Any]:
        null_count = data[column].isnull().sum()
        total = len(data)
        return {
            "Column": column,
            "Rule": "Null values",
            "Pass": total - null_count,
            "Fail": null_count,
            "Pass Rate": f"{((total - null_count) / total) * 100:.2f}%"
        }

    def _validate_uniqueness(self, data: pd.DataFrame, column: str) -> Dict[str, Any]:
        duplicates = data.duplicated(subset=[column], keep=False).sum()
        total = len(data)
        return {
            "Column": column,
            "Rule": "Unique values",
            "Pass": total - duplicates,
            "Fail": duplicates,
            "Pass Rate": f"{((total - duplicates) / total) * 100:.2f}%"
        }

    def _validate_allowed_values(self, data: pd.DataFrame, column: str, allowed_values: List[Any]) -> Dict[str, Any]:
        invalid_count = (~data[column].isin(allowed_values)).sum()
        total = len(data)
        return {
            "Column": column,
            "Rule": "Values outside allowed list",
            "Pass": total - invalid_count,
            "Fail": invalid_count,
            "Pass Rate": f"{((total - invalid_count) / total) * 100:.2f}%"
        }

    def _validate_regex(self, data: pd.DataFrame, column: str, pattern: str) -> Dict[str, Any]:
        invalid_count = (~data[column].astype(str).str.match(pattern, na=False)).sum()
        total = len(data)
        return {
            "Column": column,
            "Rule": "Values not matching regex",
            "Pass": total - invalid_count,
            "Fail": invalid_count,
            "Pass Rate": f"{((total - invalid_count) / total) * 100:.2f}%"
        }

    def _validate_range(self, data: pd.DataFrame, column: str, rules: Dict[str, Any]) -> Dict[str, Any]:
        numeric_col = pd.to_numeric(data[column], errors='coerce')
        min_val = rules.get(VRule.MIN_VALUE.value)
        max_val = rules.get(VRule.MAX_VALUE.value)
        
        invalid_mask = numeric_col.isnull()
        if min_val is not None:
            invalid_mask |= (numeric_col < float(min_val))
        if max_val is not None:
            invalid_mask |= (numeric_col > float(max_val))
            
        invalid_count = invalid_mask.sum()
        total = len(data)
        return {
            "Column": column,
            "Rule": "Values out of range",
            "Pass": total - invalid_count,
            "Fail": invalid_count,
            "Pass Rate": f"{((total - invalid_count) / total) * 100:.2f}%"
        }
