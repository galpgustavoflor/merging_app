import unittest
import pandas as pd
from utils import DataValidator
from constants import ValidationRule as VRule

class TestDataValidator(unittest.TestCase):
    def setUp(self):
        self.test_df = pd.DataFrame({
            'unique_col': [1, 2, 3, 4, 5],
            'duplicate_col': ['A', 'B', 'B', 'C', 'C'],
            'null_col': [1, None, 3, None, 5],
            'range_col': [1, 50, 100, 150, 200],
            'regex_col': ['123', 'abc', '456', 'def', '789'],
            'domain_col': ['X', 'Y', 'Z', 'W', 'V']
        })

    def test_uniqueness_validation(self):
        validation_rules = {
            'unique_col': {VRule.VALIDATE_UNIQUENESS.value: True},
            'duplicate_col': {VRule.VALIDATE_UNIQUENESS.value: True}
        }
        
        results = DataValidator.execute_validation(self.test_df, validation_rules)
        
        # Find uniqueness validation results
        unique_results = {
            result['Column Name']: result
            for result in results
            if result['Rule'] == 'Unique values'
        }
        
        # Test unique column
        self.assertEqual(
            unique_results['unique_col']['Fail'],
            0,
            "Unique column should have no failures"
        )
        self.assertEqual(
            unique_results['unique_col']['Pass'],
            5,
            "Unique column should have all passes"
        )
        
        # Test duplicate column
        self.assertEqual(
            unique_results['duplicate_col']['Fail'],
            4,  # 2 pairs of duplicates = 4 records
            "Duplicate column should have 4 failures"
        )
        self.assertEqual(
            unique_results['duplicate_col']['Pass'],
            1,  # Only one unique value
            "Duplicate column should have 1 pass"
        )

    def test_null_validation(self):
        validation_rules = {
            'null_col': {VRule.VALIDATE_NULLS.value: True}
        }
        
        results = DataValidator.execute_validation(self.test_df, validation_rules)
        null_result = next(r for r in results if r['Column Name'] == 'null_col' and r['Rule'] == 'Null values')
        
        self.assertEqual(null_result['Pass'], 3)  # 3 non-null values
        self.assertEqual(null_result['Fail'], 2)  # 2 null values

    def test_range_validation(self):
        validation_rules = {
            'range_col': {
                VRule.VALIDATE_RANGE.value: True,
                VRule.MIN_VALUE.value: 0,
                VRule.MAX_VALUE.value: 100
            }
        }
        
        results = DataValidator.execute_validation(self.test_df, validation_rules)
        range_result = next(r for r in results if r['Column Name'] == 'range_col' and r['Rule'] == 'Values out of range')
        
        self.assertEqual(range_result['Pass'], 3)  # Values within range
        self.assertEqual(range_result['Fail'], 2)  # Values outside range

    def test_regex_validation(self):
        validation_rules = {
            'regex_col': {
                VRule.VALIDATE_REGEX.value: r'^\d+$'  # Only digits
            }
        }
        
        results = DataValidator.execute_validation(self.test_df, validation_rules)
        regex_result = next(r for r in results if r['Column Name'] == 'regex_col' and r['Rule'] == 'Values not matching regex')
        
        self.assertEqual(regex_result['Pass'], 3)  # Numeric strings
        self.assertEqual(regex_result['Fail'], 2)  # Non-numeric strings

    def test_domain_validation(self):
        validation_rules = {
            'domain_col': {
                VRule.VALIDATE_LIST_OF_VALUES.value: ['X', 'Y', 'Z']
            }
        }
        
        results = DataValidator.execute_validation(self.test_df, validation_rules)
        domain_result = next(r for r in results if r['Column Name'] == 'domain_col' and r['Rule'] == 'Values outside allowed list')
        
        self.assertEqual(domain_result['Pass'], 3)  # Values in allowed list
        self.assertEqual(domain_result['Fail'], 2)  # Values outside allowed list

if __name__ == '__main__':
    unittest.main()
