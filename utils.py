import logging
import logging.config
from typing import Optional, Tuple, Dict, List
import pandas as pd
import dask.dataframe as dd
import json
import streamlit as st
from pathlib import Path
from constants import ValidationRule as VRule, FILE_TYPES, Column, Functions
from config import LOGGING_CONFIG, DASK_CONFIG, STREAMLIT_CONFIG
import numpy as np

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class FileLoader:
    @staticmethod
    @st.cache_data(ttl=3600)  # Add cache TTL
    def load_file(uploaded_file) -> Optional[pd.DataFrame]:
        try:
            # Add file size check
            max_size = STREAMLIT_CONFIG["max_file_size"] * 1024 * 1024  # Convert MB to bytes
            if uploaded_file.size > max_size:
                raise ValueError(f"File size exceeds maximum limit of {STREAMLIT_CONFIG['max_file_size']}MB")

            file_path = Path(uploaded_file.name)
            if file_path.suffix not in ['.xlsx', '.csv']:
                raise ValueError("Unsupported file format. Only .xlsx and .csv files are supported.")

            # Add content type validation
            content_type = uploaded_file.type
            if content_type not in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', 'text/csv']:
                raise ValueError("Invalid file content type")

            if file_path.suffix == '.xlsx':
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_path.suffix == '.csv':
                try:
                    df = pd.read_csv(
                        uploaded_file,
                        engine='c',
                        on_bad_lines='warn',
                        low_memory=False
                    )
                except Exception:
                    df = pd.read_csv(
                        uploaded_file,
                        engine='python',
                        on_bad_lines='skip'
                    )
            else:
                raise ValueError("Unsupported file format")
            return FileLoader._process_dataframe(df)
        except Exception as e:
            logger.error("Error loading file", exc_info=True, extra={
                'filename': uploaded_file.name,
                'file_size': uploaded_file.size,
                'content_type': uploaded_file.type
            })
            raise RuntimeError(f"Failed to load file: {str(e)}") from e

    @staticmethod
    def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.convert_dtypes()
            df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
            for col in df.columns:
                if pd.api.types.is_object_dtype(df[col]):
                    numeric_col = pd.to_numeric(df[col], errors='ignore')
                    if not pd.api.types.is_object_dtype(numeric_col):
                        df[col] = numeric_col
            return df
        except Exception as e:
            logger.error(f"Error processing dataframe: {str(e)}", exc_info=True)
            raise

def clean_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and convert DataFrame types for Arrow compatibility."""
    df = df.copy()
    
    for col in df.columns:
        # Detect Pandas nullable integer column ("Int64") and convert using a lambda to native int
        if hasattr(df[col].dtype, 'name') and df[col].dtype.name == 'Int64':
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else None)
        # Handle other integer dtypes similarly
        elif pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].apply(lambda x: int(x) if pd.notna(x) else None)
        # Convert object columns to strings
        elif df[col].dtype == 'object':
            df[col] = df[col].astype(str)
        else:
            try:
                pd.api.types.infer_dtype(df[col])
            except Exception:
                df[col] = df[col].astype(str)
    
    return df

class ConfigLoader:
    @staticmethod
    def load_json_config(file_content: str) -> Dict:
        try:
            return json.loads(file_content)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON configuration: {str(e)}")
            return {}

    @staticmethod
    def load_mapping_from_json(json_file) -> None:
        try:
            content = json_file.read()
            loaded_mapping = json.loads(content)
            st.session_state.mapping = loaded_mapping
            st.success("Mapping rules loaded successfully.")
        except Exception as e:
            st.error(f"Error loading mapping rules: {str(e)}")

    @staticmethod
    def load_validations_from_json(json_file) -> None:
        try:
            content = json_file.read()
            loaded_rules = json.loads(content)
            st.session_state.validation_rules = loaded_rules
            st.success("Validation rules loaded successfully.")
        except Exception as e:
            st.error(f"Error loading validation rules: {str(e)}")
            loaded_rules = {}
        st.write("Loaded rules:", loaded_rules)

class DataValidator:
    @staticmethod
    def execute_validation(df: pd.DataFrame, validation_rules: dict) -> List[dict]:
        st.write("Executing data validation...")
        results = []
        total_records = len(df)
        for col, rules in validation_rules.items():
            if col not in df.columns:
                continue
            # Null check
            if rules.get(VRule.VALIDATE_NULLs.value, False):
                null_count = df[col].isnull().sum()
                results.append({
                    Column.NAME.value: col,
                    "Rule": "Null values",
                    "Pass": total_records - null_count,
                    "Fail": null_count,
                    "Pass %": f"{((total_records - null_count)/total_records)*100:.2f}%" if total_records else "0.00%"
                })
            # Uniqueness check
            if rules.get(VRule.VALIDATE_UNIQUENESS.value, False):
                unique = df[col].nunique() == total_records
                fail = 0 if unique else total_records - df[col].nunique()
                results.append({
                    Column.NAME.value: col,
                    "Rule": "Unique values",
                    "Pass": total_records if unique else df[col].nunique(),
                    "Fail": fail,
                    "Pass %": "100.00%" if unique else f"{(df[col].nunique()/total_records)*100:.2f}%"
                })
            # Allowed values check
            if VRule.VALIDATE_LIST_OF_VALUES.value in rules:
                allowed = rules[VRule.VALIDATE_LIST_OF_VALUES.value]
                not_allowed = df[~df[col].isin(allowed)]
                fail_count = not_allowed.shape[0]
                results.append({
                    Column.NAME.value: col,
                    "Rule": "Values outside allowed list",
                    "Pass": total_records - fail_count,
                    "Fail": fail_count,
                    "Pass %": f"{((total_records - fail_count)/total_records)*100:.2f}%"
                })
            # Regex check
            if VRule.VALIDATE_REGEX.value in rules:
                regex = rules[VRule.VALIDATE_REGEX.value]
                mismatch = df[~df[col].astype(str).str.match(regex, na=False)]
                fail_count = mismatch.shape[0]
                results.append({
                    Column.NAME.value: col,
                    "Rule": "Values not matching regex",
                    "Pass": total_records - fail_count,
                    "Fail": fail_count,
                    "Pass %": f"{((total_records - fail_count)/total_records)*100:.2f}%"
                })
            # Range check
            if rules.get(VRule.VALIDATE_RANGE.value, False):
                try:
                    min_val = rules.get(VRule.MIN_VALUE.value)
                    max_val = rules.get(VRule.MAX_VALUE.value)
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    condition = numeric_col.notnull()
                    if min_val is not None:
                        condition &= (numeric_col >= float(min_val))
                    if max_val is not None:
                        condition &= (numeric_col <= float(max_val))
                    valid_count = condition.sum()
                    fail_count = total_records - valid_count
                    results.append({
                        Column.NAME.value: col,
                        "Rule": "Values out of range",
                        "Pass": valid_count,
                        "Fail": fail_count,
                        "Pass %": f"{(valid_count/total_records)*100:.2f}%"
                    })
                except Exception as e:
                    st.warning(f"Range validation failed for column {col}: {str(e)}")
        return results

def apply_rules(df: pd.DataFrame, mapping_config: dict) -> pd.DataFrame:
    """Apply transformation rules to the dataframe."""
    df = df.copy()
    key_source = mapping_config.get('key_source', [])
    
    # Collect all aggregations to perform them together
    agg_cols = {}
    
    # First handle conversions and prepare aggregations
    for col, rules in mapping_config.get('mappings', {}).items():
        if rules["function"] == Functions.CONVERSION.value:
            if rules.get("transformation"):
                mapping_dict = json.loads(rules["transformation"])
                df[col] = df[col].astype(str).map(mapping_dict)
                # After conversion, include in aggregation as 'first' to keep the mapped value
                agg_cols[col] = 'first'
        elif rules["function"] == Functions.DIRECT.value:
            # For direct mappings, include in aggregation as 'first'
            agg_cols[col] = 'first'
        elif rules["function"] == Functions.AGGREGATION.value:
            # Collect aggregation rules with specified function
            agg_cols[col] = rules["transformation"]
    
    # If there are any columns to aggregate
    if agg_cols:
        try:
            # Keep only necessary columns (keys + columns to aggregate)
            columns_to_keep = key_source + list(agg_cols.keys())
            df = df[columns_to_keep].copy()
            
            # Perform all aggregations at once
            df = df.groupby(key_source, as_index=False).agg(agg_cols)
            
            logger.info(f"Aggregated columns: {list(agg_cols.keys())}")
            logger.info(f"Final columns: {df.columns.tolist()}")
        except Exception as e:
            logger.error(f"Error during aggregation: {str(e)}")
            raise
    
    return df

def normalize_keys(df: pd.DataFrame, key_columns: List[str]) -> pd.DataFrame:
    """Normalize key columns for consistent matching."""
    df = df.copy()
    for col in key_columns:
        if col in df.columns:
            # Convert to string and strip whitespace
            df[col] = df[col].astype(str).str.strip()
            # Remove special characters and normalize case
            df[col] = df[col].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
            df[col] = df[col].str.lower()
    return df

def preprocess_dataframe(df: pd.DataFrame, key_columns: List[str]) -> pd.DataFrame:
    """Preprocess DataFrame for matching."""
    df = df.copy()
    # Remove completely empty rows
    df = df.dropna(how='all')
    # Fill NA in key columns with a special marker
    df[key_columns] = df[key_columns].fillna('__NA__')
    return df

def validate_merge_keys(df: pd.DataFrame, key_columns: List[str]) -> bool:
    """Validate if merge keys are unique."""
    if not key_columns:
        return False
    return df.groupby(key_columns).size().max() == 1

@st.cache_data
def execute_matching_dask(df_source: pd.DataFrame, df_target: pd.DataFrame, mapping_config: dict) -> Tuple[dd.DataFrame, dict]:
    """Execute the matching process using Dask."""
    try:
        # Get keys from mapping config
        key_source = mapping_config.get('key_source', [])
        key_target = mapping_config.get('key_target', [])
        
        if not key_source or not key_target:
            raise ValueError("Source and target keys must be defined")
        
        # Get mapped columns
        mapped_columns = {}
        for col, rules in mapping_config.get('mappings', {}).items():
            if rules.get('destinations'):
                # Map source column to its target destination(s)
                for dest in rules['destinations']:
                    mapped_columns[col] = dest
        
        # Combine keys and mapped columns for merging
        merge_on_source = key_source + list(mapped_columns.keys())
        merge_on_target = key_target + list(mapped_columns.values())
        
        # Convert nullable types in key columns
        def convert_nullable_types(df):
            for col in df.columns:
                if str(df[col].dtype).startswith('Int'):
                    df[col] = df[col].astype('float').astype('Int64').fillna(-999999).astype('int64')
            return df
        
        df_source = convert_nullable_types(df_source.copy())
        df_target = convert_nullable_types(df_target.copy())
        
        # Apply mapping rules
        df_source = apply_rules(df_source, mapping_config)
        
        # Convert to Dask DataFrames
        ddf_source = dd.from_pandas(df_source, npartitions=DASK_CONFIG["npartitions"])
        ddf_target = dd.from_pandas(df_target, npartitions=DASK_CONFIG["npartitions"])
        
        # Perform merge with all matching columns
        ddf_merged = ddf_source.merge(
            ddf_target,
            left_on=merge_on_source,
            right_on=merge_on_target,
            how='outer',
            indicator=True
        )
        
        # Compute statistics
        stats = {
            'total_match': len(ddf_merged[ddf_merged['_merge'] == 'both'].compute()),
            'missing_source': len(ddf_merged[ddf_merged['_merge'] == 'right_only'].compute()),
            'missing_target': len(ddf_merged[ddf_merged['_merge'] == 'left_only'].compute())
        }
        
        return ddf_merged, stats
        
    except Exception as e:
        logger.error(f"Error in execute_matching_dask: {e}", exc_info=True)
        raise

def handle_large_file(file_path: str, chunk_size: int = 10000) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
