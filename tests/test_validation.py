import pytest
from app import execute_data_validation
from constants import ValidationRule as VRule

def test_validate_nulls(sample_target_df):
    """Test null value validation."""
    # Add some null values
    df = sample_target_df.copy()
    df.loc[0, 'Nome_Destino'] = None
    
    rules = {
        'Nome_Destino': {
            VRule.VALIDATE_NULLS.value: True
        }
    }
    
    results = execute_data_validation(df, rules)
    assert len(results) == 1
    assert results[0]['Fail'] == 1

def test_validate_range(sample_target_df):
    """Test range validation."""
    rules = {
        'Idade_Destino': {
            VRule.VALIDATE_RANGE.value: True,
            VRule.MIN_VALUE.value: 30,
            VRule.MAX_VALUE.value: 40
        }
    }
    
    results = execute_data_validation(sample_target_df, rules)
    assert len(results) == 1
    assert results[0]['Fail'] > 0  # Some values should be outside range

def test_validate_unique(sample_target_df):
    """Test uniqueness validation."""
    # Add duplicate value
    df = sample_target_df.copy()
    df.loc[5, 'Nome_Destino'] = df.loc[0, 'Nome_Destino']
    
    rules = {
        'Nome_Destino': {
            VRule.VALIDATE_UNIQUENESS.value: True
        }
    }
    
    results = execute_data_validation(df, rules)
    assert len(results) == 1
    assert results[0]['Fail'] == 2  # Two records involved in duplication
