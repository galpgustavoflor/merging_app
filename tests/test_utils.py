import pytest
from utils import FileLoader, ConfigLoader, DataValidator
import pandas as pd
import streamlit as st

def test_file_loader_csv(sample_files):
    """Test loading CSV files."""
    with open(sample_files['source_csv'], 'rb') as f:
        df = FileLoader.load_file(f)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    assert list(df.columns) == ['ID', 'Name', 'Age', 'Value']

def test_file_loader_invalid_format():
    """Test loading invalid file format."""
    class MockFile:
        name = "test.txt"
    
    df = FileLoader.load_file(MockFile())
    assert df is None

def test_data_validator_duplicate_columns(sample_source_df):
    """Test handling of duplicate column names."""
    # Create DataFrame with duplicate columns
    df = sample_source_df.copy()
    df.columns = ['ID', 'ID', 'Age', 'Value']
    
    df_cleaned, renamed = DataValidator.validate_column_names(df)
    assert list(df_cleaned.columns) == ['ID', 'ID.1', 'Age', 'Value']
    assert len(renamed) == 1

def test_config_loader_valid_json(sample_files):
    """Test loading valid JSON configuration."""
    with open(sample_files['mapping_json'], 'r') as f:
        config = ConfigLoader.load_json_config(f.read())
    assert isinstance(config, dict)
    assert 'key_source' in config
    assert 'mappings' in config
