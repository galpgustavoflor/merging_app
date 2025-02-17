import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

@pytest.fixture
def sample_source_df():
    """Create a sample source dataframe for testing."""
    return pd.DataFrame({
        'ID': range(1, 6),
        'Name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'Age': [25, 30, 35, 40, 45],
        'Value': [100.0, 200.0, 300.0, 400.0, 500.0]
    })

@pytest.fixture
def sample_target_df():
    """Create a sample target dataframe for testing."""
    return pd.DataFrame({
        'ID_Destino': range(1, 7),
        'Nome_Destino': ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'David'],
        'Idade_Destino': [25, 30, 35, 40, 45, 50],
        'Valor_Destino': [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    })

@pytest.fixture
def sample_validation_rules():
    """Create sample validation rules for testing."""
    return {
        'ID_Destino': {
            'validate_nulls': True,
            'validate_uniqueness': True
        },
        'Nome_Destino': {
            'validate_list_of_values': ['John', 'Jane', 'Bob', 'Alice', 'Charlie']
        },
        'Idade_Destino': {
            'validate_range': True,
            'min_value': 20,
            'max_value': 60
        }
    }

@pytest.fixture
def sample_mapping_config():
    """Create sample mapping configuration for testing."""
    return {
        'key_source': ['ID'],
        'key_target': ['ID_Destino'],
        'mappings': {
            'Name': {
                'destinations': ['Nome_Destino'],
                'function': 'Direct Match',
                'transformation': None
            },
            'Age': {
                'destinations': ['Idade_Destino'],
                'function': 'Conversion',
                'transformation': {'25': '20-30', '30': '30-40', '35': '30-40'}
            }
        }
    }

@pytest.fixture
def sample_files(sample_source_df, sample_target_df, sample_validation_rules, sample_mapping_config):
    """Create temporary test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create CSV files
        source_path = Path(tmpdir) / "source.csv"
        target_path = Path(tmpdir) / "target.csv"
        sample_source_df.to_csv(source_path, index=False)
        sample_target_df.to_csv(target_path, index=False)
        
        # Create JSON files
        validation_path = Path(tmpdir) / "validation_rules.json"
        mapping_path = Path(tmpdir) / "mapping_config.json"
        with open(validation_path, 'w') as f:
            json.dump(sample_validation_rules, f)
        with open(mapping_path, 'w') as f:
            json.dump(sample_mapping_config, f)
        
        yield {
            'source_csv': source_path,
            'target_csv': target_path,
            'validation_json': validation_path,
            'mapping_json': mapping_path
        }
