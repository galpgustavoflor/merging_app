import logging
import logging.config
from typing import Optional, Tuple, Dict, Union
import pandas as pd
import dask.dataframe as dd
import json
import streamlit as st
from pathlib import Path
from functools import lru_cache
from constants import ValidationRule as VRule
from config import LOGGING_CONFIG, DASK_CONFIG
import numpy as np

# Configure logging
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class FileLoader:
    @staticmethod
    @st.cache_data
    def load_file(uploaded_file) -> Optional[pd.DataFrame]:
        """Load and process an uploaded file with caching."""
        try:
            file_path = Path(uploaded_file.name)
            if file_path.suffix == '.xlsx':
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_path.suffix == '.csv':
                # First try with c engine for better performance
                try:
                    df = pd.read_csv(
                        uploaded_file,
                        engine='c',
                        on_bad_lines='warn',
                        low_memory=False
                    )
                except:
                    # Fallback to python engine if c engine fails
                    df = pd.read_csv(
                        uploaded_file,
                        engine='python',
                        on_bad_lines='skip'
                    )
            else:
                raise ValueError("Unsupported file format")
            
            return FileLoader._process_dataframe(df)
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}", exc_info=True)
            st.error(f"Error loading file: {str(e)}")
            return None

    @staticmethod
    def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and optimize dataframe."""
        try:
            # Optimize memory usage
            df = df.convert_dtypes()
            
            # Clean column names
            df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
            
            # Optimize numeric columns
            for col in df.columns:
                if pd.api.types.is_object_dtype(df[col]):
                    # Try converting to numeric, keeping original if fails
                    numeric_col = pd.to_numeric(df[col], errors='ignore')
                    if not pd.api.types.is_object_dtype(numeric_col):
                        df[col] = numeric_col
            
            return df
        except Exception as e:
            logger.error(f"Error processing dataframe: {str(e)}", exc_info=True)
            raise

class ConfigLoader:
    @staticmethod
    def load_json_config(file_content: str) -> Dict:
        """Load and validate JSON configuration."""
        try:
            return json.loads(file_content)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON configuration: {str(e)}")
            return {}

    @staticmethod
    def load_mapping_from_json(json_file) -> None:
        """Load mapping configuration from JSON file."""
        try:
            content = json_file.read()
            loaded_mapping = json.loads(content)
            st.session_state.mapping = loaded_mapping
            st.success("Mapping rules loaded successfully.")
        except Exception as e:
            st.error(f"Error loading mapping rules: {str(e)}")

    @staticmethod
    def load_validations_from_json(json_file) -> None:
        """Load validation rules from JSON file."""
        try:
            content = json_file.read()
            loaded_rules = json.loads(content)
            st.session_state.validation_rules = loaded_rules
            st.success("Validation rules loaded successfully.")
        except Exception as e:
            st.error(f"Error loading validation rules: {str(e)}")
            loaded_rules = None  # Ensure loaded_rules is defined
        st.write("Loaded rules:", loaded_rules)

class DataValidator:
    @staticmethod
    def validate_column_names(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Validate and fix duplicate column names."""
        renamed_columns = []
        seen_columns = {}
        new_columns = []

        for col in df.columns:
            if (col in seen_columns):
                seen_columns[col] += 1
                new_name = f"{col}.{seen_columns[col]}"
                renamed_columns.append((col, new_name))
                new_columns.append(new_name)
            else:
                seen_columns[col] = 1
                new_columns.append(col)
        
        df.columns = new_columns
        renamed_df = pd.DataFrame(renamed_columns, columns=["Original", "New"])
        return df, renamed_df

def compute_statistics(df: pd.DataFrame) -> dict:
    """Compute statistics for a dataframe."""
    if df.empty:
        return {
            'count': 0,
            'mean': 0,
            'sum': 0
        }
    return {
        'count': len(df),
        'mean': df['amount'].mean() if 'amount' in df.columns else 0,
        'sum': df['amount'].sum() if 'amount' in df.columns else 0
    }

def handle_large_file(file_path: str, chunk_size: int = 10000) -> pd.DataFrame:
    """Handle large file loading with chunking."""
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks) if chunks else pd.DataFrame()

def validate_unique(df, column):
    """Validate uniqueness of values in a column."""
    duplicates = df[column].duplicated().sum()
    return duplicates == 0
