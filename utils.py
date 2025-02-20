import logging
import logging.config
from typing import Optional, Tuple, Dict, List
import pandas as pd
import dask.dataframe as dd
import json
import streamlit as st
from pathlib import Path
from constants import ValidationRule as VRule, FILE_TYPES, Column, Functions
from config import LOGGING_CONFIG, DASK_CONFIG
import numpy as np

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class FileLoader:
    @staticmethod
    @st.cache_data
    def load_file(uploaded_file) -> Optional[pd.DataFrame]:
        try:
            file_path = Path(uploaded_file.name)
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
            logger.error(f"Error loading file: {str(e)}", exc_info=True)
            st.error(f"Error loading file: {str(e)}")
            return None

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

@st.cache_data
def execute_matching_dask(df_source: pd.DataFrame, df_target: pd.DataFrame, key_source: List[str], key_target: List[str]) -> Tuple[dd.DataFrame, Dict[str, int]]:
    npartitions = DASK_CONFIG["default_npartitions"]
    ddf_source = dd.from_pandas(df_source, npartitions=npartitions)
    ddf_target = dd.from_pandas(df_target, npartitions=npartitions)
    ddf_merged = ddf_source.merge(ddf_target, left_on=key_source, right_on=key_target, how="outer", indicator=True)
    stats = {
        "total_match": int(ddf_merged[ddf_merged['_merge'] == 'both'].shape[0].compute()),
        "missing_source": int(ddf_merged[ddf_merged['_merge'] == 'right_only'].shape[0].compute()),
        "missing_target": int(ddf_merged[ddf_merged['_merge'] == 'left_only'].shape[0].compute())
    }
    return ddf_merged, stats

def handle_large_file(file_path: str, chunk_size: int = 10000) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
