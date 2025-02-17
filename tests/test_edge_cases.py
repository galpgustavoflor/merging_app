import pytest
import pandas as pd
import numpy as np
from utils import FileLoader, ConfigLoader
from constants import ValidationRule as VRule

def test_empty_dataframe_handling(sample_validation_rules):
    """Test validation with empty dataframe."""
    empty_df = pd.DataFrame(columns=['col1', 'col2'])
    from app import execute_data_validation
    
    results = execute_data_validation(empty_df, sample_validation_rules)
    assert isinstance(results, list)
    assert len(results) > 0
    for result in results:
        assert result['Pass'] == 0

def test_mixed_type_column_handling(sample_target_df):
    """Test handling of mixed data types in columns."""
    df = sample_target_df.copy()
    df.loc[0, 'Idade_Destino'] = 'invalid'
    
    rules = {
        'Idade_Destino': {
            VRule.VALIDATE_RANGE.value: True,
            VRule.MIN_VALUE.value: 0,
            VRule.MAX_VALUE.value: 100
        }
    }
    
    from app import execute_data_validation
    results = execute_data_validation(df, rules)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]['Fail'] > 0

def test_large_file_handling():
    """Test handling of large DataFrames."""
    large_df = pd.DataFrame({
        'ID': range(100000),
        'Value': np.random.rand(100000)
    })
    
    # Test file loading
    import io
    csv_buffer = io.StringIO()
    large_df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    
    class MockFile:
        name = "large.csv"
        def read(self):
            return csv_buffer.read()
    
    loaded_df = FileLoader.load_file(MockFile())
    assert isinstance(loaded_df, pd.DataFrame)
    assert len(loaded_df) == 100000

def test_invalid_json_handling():
    """Test handling of invalid JSON configurations."""
    invalid_json = "{'invalid': json}"
    config = ConfigLoader.load_json_config(invalid_json)
    assert isinstance(config, dict)
    assert len(config) == 0

def test_memory_efficient_processing(sample_source_df, sample_target_df):
    """Test memory-efficient processing of data matching."""
    from app import execute_matching_dask
    import psutil
    import os
    
    # Monitor memory usage
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    
    # Execute matching
    ddf_final, stats = execute_matching_dask(
        sample_source_df,
        sample_target_df,
        ['ID'],
        ['ID_Destino']
    )
    
    mem_after = process.memory_info().rss
    mem_increase = (mem_after - mem_before) / 1024 / 1024  # MB
    
    # Memory increase should be reasonable
    assert mem_increase < 100  # Less than 100MB increase
