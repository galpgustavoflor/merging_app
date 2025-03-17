import logging
import logging.config
from typing import Optional, Tuple, Dict, List, Union, Any
from abc import ABC, abstractmethod
import pandas as pd
import dask.dataframe as dd
import dask
import json
import streamlit as st
from pathlib import Path
from constants import ValidationRule as VRule, FILE_TYPES, Column, Functions
from config import LOGGING_CONFIG, DASK_CONFIG, STREAMLIT_CONFIG
import numpy as np
import yaml
import re
import unicodedata

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

class DataLoader(ABC):
    """Abstract base class for data loading operations."""
    
    @abstractmethod
    def load(self, source) -> Union[pd.DataFrame, dd.DataFrame]:
        """Load data from a source into a DataFrame."""
        pass

    @staticmethod
    def _process_dataframe(df: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame]:
        """Common DataFrame processing logic."""
        try:
            if isinstance(df, dd.DataFrame):
                df = df.map_partitions(lambda pdf: pdf.convert_dtypes())
                df.columns = [col.strip().replace("\ufeff", "") for col in df.columns]
                return df
            else:
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

class FileLoader(DataLoader):
    """Handles loading data from file sources."""
    
    @staticmethod
    @st.cache_data(ttl=3600)
    def load_file(uploaded_file) -> Union[pd.DataFrame, dd.DataFrame]:
        """Maintained for backward compatibility."""
        loader = FileLoader()
        return loader.load(uploaded_file)

    @staticmethod
    def _should_use_dask(file_size: int) -> bool:
        return file_size > (500 * 1024 * 1024)

    @staticmethod
    def _detect_delimiter(file_obj, num_lines=5) -> str:
        """Detect the delimiter in a CSV file by analyzing the first few lines."""
        try:
            # Store current position
            pos = file_obj.tell()
            
            # Read sample lines
            sample_lines = []
            for _ in range(num_lines):
                line = file_obj.readline().decode('utf-8')
                if line:
                    sample_lines.append(line)
            
            # Reset file position
            file_obj.seek(pos)
            
            # Common delimiters to check
            delimiters = [',', ';', '|', '\t']
            
            # Count occurrences of each delimiter
            delimiter_counts = {d: sum(line.count(d) for line in sample_lines) for d in delimiters}
            
            # Get the most common delimiter
            max_count = max(delimiter_counts.values())
            if max_count > 0:
                most_common = [d for d, count in delimiter_counts.items() if count == max_count][0]
                logger.info(f"Detected delimiter: '{most_common}'")
                return most_common
                
            return ','  # Default to comma if no clear delimiter is found
            
        except Exception as e:
            logger.warning(f"Error detecting delimiter: {str(e)}. Using default ','")
            return ','

    def load(self, uploaded_file) -> Union[pd.DataFrame, dd.DataFrame]:
        """Implementation of abstract load method."""
        try:
            max_size = STREAMLIT_CONFIG["max_file_size"] * 1024 * 1024
            if uploaded_file.size > max_size:
                raise ValueError(f"File size exceeds maximum limit of {STREAMLIT_CONFIG['max_file_size']}MB")

            file_path = Path(uploaded_file.name)
            if file_path.suffix not in ['.xlsx', '.csv']:
                raise ValueError("Unsupported file format. Only .xlsx and .csv files are supported.")

            use_dask = FileLoader._should_use_dask(uploaded_file.size)
            
            if file_path.suffix == '.xlsx':
                return self._load_excel(uploaded_file, use_dask)
            else:
                return self._load_csv(uploaded_file, use_dask)

        except Exception as e:
            logger.error("Error loading file", exc_info=True, extra={
                'filename': uploaded_file.name,
                'file_size': uploaded_file.size,
                'content_type': uploaded_file.type
            })
            raise RuntimeError(f"Failed to load file: {str(e)}") from e

    def _load_excel(self, file, use_dask: bool) -> Union[pd.DataFrame, dd.DataFrame]:
        """Handle Excel file loading."""
        df = pd.read_excel(file, engine='openpyxl')
        df = self._process_dataframe(df)
        if use_dask:
            return dd.from_pandas(df, npartitions=DASK_CONFIG["npartitions"])
        return df

    def _load_csv(self, file, use_dask: bool) -> Union[pd.DataFrame, dd.DataFrame]:
        """Handle CSV file loading."""
        delimiter = self._detect_delimiter(file)
        file.seek(0)

        if use_dask:
            return dd.read_csv(
                file,
                sep=delimiter,
                encoding='utf-8',
                encoding_errors='replace',
                on_bad_lines='warn'
            )
        
        try:
            df = pd.read_csv(
                file,
                sep=delimiter,
                encoding='utf-8',
                encoding_errors='replace',
                engine='c',
                on_bad_lines='warn',
                low_memory=False
            )
        except Exception as e:
            logger.warning(f"Failed to read with 'c' engine: {str(e)}. Trying 'python' engine...")
            file.seek(0)
            df = pd.read_csv(
                file,
                sep=delimiter,
                encoding='utf-8',
                encoding_errors='replace',
                engine='python',
                on_bad_lines='skip'
            )
        
        return self._process_dataframe(df)

class DatabaseLoader(DataLoader):
    """Handles loading data from database sources."""
    
    def __init__(self, connection_params: Dict[str, str]):
        self.connection_params = connection_params

    def load(self, query: str) -> Union[pd.DataFrame, dd.DataFrame]:
        """Load data from a database using SQL query."""
        # Implementation for database loading
        # This is a placeholder for future implementation
        raise NotImplementedError("Database loading not yet implemented")

class APILoader(DataLoader):
    """Handles loading data from API endpoints."""
    
    def __init__(self, api_config: Dict[str, str]):
        self.api_config = api_config

    def load(self, endpoint: str) -> Union[pd.DataFrame, dd.DataFrame]:
        """Load data from an API endpoint."""
        # Implementation for API data loading
        # This is a placeholder for future implementation
        raise NotImplementedError("API loading not yet implemented")

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
    def execute_validation(df: Union[pd.DataFrame, dd.DataFrame],
                       validation_rules: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute data validation on a DataFrame (Pandas or Dask) based on provided validation rules.

        This function iterates over the specified columns and applies a series of validations:
        - Null values check
        - Uniqueness check
        - Allowed values check
        - Regex pattern match check
        - Range check

        For Dask DataFrames, validations are batched and computed in a single call to minimize scheduler overhead.
        For Pandas DataFrames, validations are computed directly.

        Parameters:
            df (Union[pd.DataFrame, dd.DataFrame]): The DataFrame to validate.
            validation_rules (Dict[str, Dict[str, Any]]): A dictionary mapping column names to a dictionary
                of validation rules. Each rule is keyed by a validation rule identifier (e.g., VRule.VALIDATE_NULLs.value)
                and contains the necessary parameters.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with the validation results for each column and rule.
                Each dictionary includes:
                - Column name
                - Rule name
                - Number of passing records
                - Number of failing records
                - Percentage of passing records formatted as a string

        Raises:
            Warning via streamlit (st.warning) if any range validation fails.
        """
        st.write("Executing data validation...")
        results: List[Dict[str, Any]] = []

        # Compute total_records once based on the type of DataFrame.
        if isinstance(df, dd.DataFrame):
            total_records = df.map_partitions(len).sum().compute()
        else:
            total_records = len(df)

        # Iterate over each column and its associated validation rules.
        for col, rules in validation_rules.items():
            if col not in df.columns:
                continue  # Skip columns not present in the DataFrame

            # --- Dask DataFrame branch ---
            if isinstance(df, dd.DataFrame):
                # Lists to collect delayed tasks and corresponding rule identifiers
                delayed_tasks = []
                task_identifiers = []

                # Null values check
                if rules.get(VRule.VALIDATE_NULLs.value, False):
                    null_task = df[col].isnull().sum()  # Delayed Dask expression
                    delayed_tasks.append(null_task)
                    task_identifiers.append("Null values")

                # Uniqueness check
                if rules.get(VRule.VALIDATE_UNIQUENESS.value, False):
                    nunique_task = df[col].nunique()
                    delayed_tasks.append(nunique_task)
                    task_identifiers.append("Unique values")

                # Allowed values check
                if VRule.VALIDATE_LIST_OF_VALUES.value in rules:
                    allowed = rules[VRule.VALIDATE_LIST_OF_VALUES.value]
                    not_allowed_task = df[~df[col].isin(allowed)].map_partitions(len).sum()
                    delayed_tasks.append(not_allowed_task)
                    task_identifiers.append("Values outside allowed list")

                # Regex check
                if VRule.VALIDATE_REGEX.value in rules:
                    regex = rules[VRule.VALIDATE_REGEX.value]
                    mismatch_task = df[~df[col].astype(str).str.match(regex, na=False)] \
                                    .map_partitions(len).sum()
                    delayed_tasks.append(mismatch_task)
                    task_identifiers.append("Values not matching regex")

                # Range check
                if rules.get(VRule.VALIDATE_RANGE.value, False):
                    try:
                        min_val = rules.get(VRule.MIN_VALUE.value)
                        max_val = rules.get(VRule.MAX_VALUE.value)
                        numeric_col = dd.to_numeric(df[col], errors='coerce')
                        condition = numeric_col.notnull()
                        if min_val is not None:
                            condition = condition & (numeric_col >= float(min_val))
                        if max_val is not None:
                            condition = condition & (numeric_col <= float(max_val))
                        valid_task = condition.sum()
                        delayed_tasks.append(valid_task)
                        task_identifiers.append("Values out of range")
                    except Exception as e:
                        st.warning(f"Range validation failed for column {col}: {str(e)}")

                # Execute all delayed tasks in one batch
                computed_results = dask.compute(*delayed_tasks) if delayed_tasks else []
                task_index = 0  # To track the order of computed results

                # Map each computed result to its corresponding rule and build the result dictionary
                for rule_type in task_identifiers:
                    if rule_type == "Null values":
                        null_count = computed_results[task_index]
                        results.append({
                            Column.NAME.value: col,
                            "Rule": "Null values",
                            "Pass": total_records - null_count,
                            "Fail": null_count,
                            "Pass %": f"{((total_records - null_count) / total_records) * 100:.2f}%" 
                                    if total_records else "0.00%"
                        })
                        task_index += 1

                    elif rule_type == "Unique values":
                        nunique_count = computed_results[task_index]
                        unique = (nunique_count == total_records)
                        fail = 0 if unique else total_records - nunique_count
                        results.append({
                            Column.NAME.value: col,
                            "Rule": "Unique values",
                            "Pass": total_records if unique else nunique_count,
                            "Fail": fail,
                            "Pass %": "100.00%" if unique else f"{(nunique_count / total_records) * 100:.2f}%"
                        })
                        task_index += 1

                    elif rule_type == "Values outside allowed list":
                        fail_count = computed_results[task_index]
                        results.append({
                            Column.NAME.value: col,
                            "Rule": "Values outside allowed list",
                            "Pass": total_records - fail_count,
                            "Fail": fail_count,
                            "Pass %": f"{((total_records - fail_count) / total_records) * 100:.2f}%"
                        })
                        task_index += 1

                    elif rule_type == "Values not matching regex":
                        fail_count = computed_results[task_index]
                        results.append({
                            Column.NAME.value: col,
                            "Rule": "Values not matching regex",
                            "Pass": total_records - fail_count,
                            "Fail": fail_count,
                            "Pass %": f"{((total_records - fail_count) / total_records) * 100:.2f}%"
                        })
                        task_index += 1

                    elif rule_type == "Values out of range":
                        valid_count = computed_results[task_index]
                        fail_count = total_records - valid_count
                        results.append({
                            Column.NAME.value: col,
                            "Rule": "Values out of range",
                            "Pass": valid_count,
                            "Fail": fail_count,
                            "Pass %": f"{(valid_count / total_records) * 100:.2f}%"
                        })
                        task_index += 1

            # --- Pandas DataFrame branch ---
            else:
                # Null values check
                if rules.get(VRule.VALIDATE_NULLs.value, False):
                    null_count = df[col].isnull().sum()
                    results.append({
                        Column.NAME.value: col,
                        "Rule": "Null values",
                        "Pass": total_records - null_count,
                        "Fail": null_count,
                        "Pass %": f"{((total_records - null_count) / total_records) * 100:.2f}%" 
                                if total_records else "0.00%"
                    })

                # Uniqueness check
                if rules.get(VRule.VALIDATE_UNIQUENESS.value, False):
                    nunique_count = df[col].nunique()
                    unique = (nunique_count == total_records)
                    fail = 0 if unique else total_records - nunique_count
                    results.append({
                        Column.NAME.value: col,
                        "Rule": "Unique values",
                        "Pass": total_records if unique else nunique_count,
                        "Fail": fail,
                        "Pass %": "100.00%" if unique else f"{(nunique_count / total_records) * 100:.2f}%"
                    })

                # Allowed values check
                if VRule.VALIDATE_LIST_OF_VALUES.value in rules:
                    allowed = rules[VRule.VALIDATE_LIST_OF_VALUES.value]
                    fail_count = df[~df[col].isin(allowed)].shape[0]
                    results.append({
                        Column.NAME.value: col,
                        "Rule": "Values outside allowed list",
                        "Pass": total_records - fail_count,
                        "Fail": fail_count,
                        "Pass %": f"{((total_records - fail_count) / total_records) * 100:.2f}%"
                    })

                # Regex check
                if VRule.VALIDATE_REGEX.value in rules:
                    regex = rules[VRule.VALIDATE_REGEX.value]
                    fail_count = df[~df[col].astype(str).str.match(regex, na=False)].shape[0]
                    results.append({
                        Column.NAME.value: col,
                        "Rule": "Values not matching regex",
                        "Pass": total_records - fail_count,
                        "Fail": fail_count,
                        "Pass %": f"{((total_records - fail_count) / total_records) * 100:.2f}%"
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
                            "Pass %": f"{(valid_count / total_records) * 100:.2f}%"
                        })
                    except Exception as e:
                        st.warning(f"Range validation failed for column {col}: {str(e)}")
                        
        return results

def _convert_using_dict(series: pd.Series, mapping_dict: dict) -> pd.Series:
    """Helper function to convert values using a mapping dictionary."""
    return series.astype(str).map(mapping_dict).fillna(series)

def apply_rules(df: Union[pd.DataFrame, dd.DataFrame], mapping_config: dict) -> Union[pd.DataFrame, dd.DataFrame]:
    """Apply transformation rules to the dataframe."""
    is_dask = isinstance(df, dd.DataFrame)
    
    # Create a deep copy to preserve all transformations
    df = df.copy() if not is_dask else df
    
    key_source = mapping_config.get('key_source', [])
    
    # Collect all aggregations to perform them together
    agg_cols = {}
    
    # First handle conversions and prepare aggregations
    for col, rules in mapping_config.get('mappings', {}).items():
        # Skip if column doesn't exist in the dataframe
        if col not in df.columns:
            logger.warning(f"Column '{col}' specified in mapping rules not found in dataframe")
            continue
            
        if rules["function"] == Functions.CONVERSION.value:
            if rules.get("transformation"):
                try:
                    mapping_dict = json.loads(rules["transformation"])
                    if is_dask:
                        # For Dask DataFrame, use map_partitions with explicit function
                        df[col] = df[col].map_partitions(_convert_using_dict, mapping_dict)
                    else:
                        df[col] = _convert_using_dict(df[col], mapping_dict)
                    agg_cols[col] = 'first'
                    logger.info(f"Applied conversion to column '{col}'")
                except Exception as e:
                    logger.error(f"Error applying conversion to '{col}': {e}")
        elif rules["function"] == Functions.DIRECT.value:
            agg_cols[col] = 'first'
        elif rules["function"] == Functions.AGGREGATION.value:
            agg_cols[col] = rules["transformation"]
    
    # If there are any columns to aggregate
    if agg_cols:
        try:
            # Keep only necessary columns
            columns_to_keep = key_source + list(agg_cols.keys())
            
            # Filter to only include columns that actually exist in the dataframe
            columns_to_keep = [col for col in columns_to_keep if col in df.columns]
            
            df = df[columns_to_keep].copy()
            
            if is_dask:
                # For Dask DataFrame, compute aggregations
                grouped = df.groupby(key_source)
                results = []
                
                # Process each column's aggregation separately
                for col, agg_func in agg_cols.items():
                    if col not in df.columns:
                        logger.warning(f"Column '{col}' not found in dataframe, skipping aggregation")
                        continue
                        
                    if agg_func == 'first':
                        # Special handling for 'first' aggregation
                        agg_result = grouped[col].first().reset_index()
                    else:
                        agg_result = grouped[col].agg(agg_func).reset_index()
                    results.append(agg_result)
                
                # Merge results if we have multiple columns
                if results:
                    df = results[0]
                    for i, other_df in enumerate(results[1:]):
                        # Use explicit merge to avoid ambiguous Series comparison
                        on_columns = key_source
                        df = df.merge(other_df, on=on_columns)
            else:
                # For Pandas DataFrame
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

# Add these helper functions at module level (outside any other functions)
def strip_strings(s):
    """
    Strip all kinds of whitespace from strings in a Series - 
    handles Unicode whitespace characters and other invisible characters.
    """
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        # Convert to string to handle numeric types safely
        return s.astype(str).apply(
            lambda x: re.sub(r'^\s+|\s+$', '', 
                            unicodedata.normalize('NFKC', x)) if isinstance(x, str) else x
        )
    return s

def clean_key_column(s):
    """
    More thorough cleaning of key columns - handles Unicode whitespace, 
    non-breaking spaces, zero-width spaces, and other invisible characters.
    """
    if pd.api.types.is_string_dtype(s) or pd.api.types.is_object_dtype(s):
        # Convert to string, normalize Unicode, and strip all whitespace types
        return s.astype(str).apply(
            lambda x: re.sub(r'^\s+|\s+$', '', 
                            unicodedata.normalize('NFKC', x)) if isinstance(x, str) else x
        )
    return s.astype(str)

@st.cache_resource
def execute_matching_dask(df_source: Union[pd.DataFrame, dd.DataFrame], 
                         df_target: Union[pd.DataFrame, dd.DataFrame], 
                         mapping_config: dict) -> Tuple[dd.DataFrame, dict]:
    """Execute the matching process using Dask."""
    try:
        # Log dataset info to confirm we're using transformed data
        logger.info(f"Source dataset info before mapping: {df_source.shape[0]} rows, {df_source.shape[1]} columns")
        logger.info(f"Target dataset info: {df_target.shape[0]} rows, {df_target.shape[1]} columns")
        
        # Log a sample of the source data to verify transformations
        if isinstance(df_source, pd.DataFrame) and len(df_source) > 0:
            sample = df_source.iloc[0].to_dict()
            logger.debug(f"Source sample before conversion to Dask: {sample}")
        
        # Convert to Dask DataFrame if input is Pandas DataFrame
        # Important: make deep copies to avoid any reference issues
        if isinstance(df_source, pd.DataFrame):
            # Ensure we're working with a clean copy
            source_copy = df_source.copy(deep=True)
            df_source = dd.from_pandas(source_copy, npartitions=DASK_CONFIG["npartitions"])
            logger.debug("Converted source to Dask DataFrame")
        if isinstance(df_target, pd.DataFrame):
            # Ensure we're working with a clean copy
            target_copy = df_target.copy(deep=True)
            df_target = dd.from_pandas(target_copy, npartitions=DASK_CONFIG["npartitions"])
            logger.debug("Converted target to Dask DataFrame")
            
        # Get keys and prepare for merge
        key_source = mapping_config.get('key_source', [])
        key_target = mapping_config.get('key_target', [])
        
        if not key_source or not key_target:
            raise ValueError("Source and target keys must be defined")
            
        # Log key columns before applying rules
        logger.debug(f"Key source columns before mapping: {key_source}")
        if len(key_source) > 0 and key_source[0] in df_source.columns:
            # Get a sample from the first key column
            try:
                # FIX: Check if the column is actually a Dask Series before calling compute()
                sample_col = df_source[key_source[0]].head()
                if hasattr(sample_col, 'compute'):
                    sample_key_values = sample_col.compute().tolist()
                else:
                    sample_key_values = sample_col.tolist()  # Regular pandas Series
                logger.debug(f"Sample key values before mapping: {sample_key_values}")
            except Exception as e:
                logger.error(f"Failed to get sample key values: {e}")
        
        # Apply mapping rules to source DataFrame
        df_source_mapped = apply_rules(df_source, mapping_config)
        logger.info(f"Applied mapping rules to source dataset: {df_source_mapped.columns.tolist()}")
        
        # Log key columns after applying rules
        if len(key_source) > 0 and key_source[0] in df_source_mapped.columns:
            # Get a sample from the first key column
            try:
                # FIX: Check if the column is actually a Dask Series before calling compute()
                sample_col = df_source_mapped[key_source[0]].head()
                if hasattr(sample_col, 'compute'):
                    sample_key_values = sample_col.compute().tolist()
                else:
                    sample_key_values = sample_col.tolist()  # Regular pandas Series
                logger.debug(f"Sample key values after mapping: {sample_key_values}")
            except Exception as e:
                logger.error(f"Failed to get sample key values after mapping: {e}")
        
        # Log data types before merge for debugging
        source_dtypes = {col: str(df_source_mapped[col].dtype) for col in key_source if col in df_source_mapped.columns}
        target_dtypes = {col: str(df_target[col].dtype) for col in key_target}
        
        # Check for data type mismatches
        mismatches = []
        for s_col, t_col in zip(key_source, key_target):
            if s_col in source_dtypes and t_col in target_dtypes:
                if source_dtypes[s_col] != target_dtypes[t_col]:
                    mismatches.append({
                        'columns': f"({s_col}, {t_col})",
                        'left_dtype': source_dtypes[s_col],
                        'right_dtype': target_dtypes[t_col]
                    })
        
        if mismatches:
            # Create user-friendly warning message
            warning_msg = (
                "⚠️ Data Type Mismatch Warning:\n\n"
                "The following columns have different data types between source and target files:\n\n"
            )
            for mismatch in mismatches:
                warning_msg += (
                    f"• Columns: {mismatch['columns']}\n"
                    f"  - Source type: {mismatch['left_dtype']}\n"
                    f"  - Target type: {mismatch['right_dtype']}\n"
                )
            warning_msg += "\nThe merge will proceed but results may be unexpected. Consider standardizing data types in your source files."
            
            st.warning(warning_msg)
        
        # Ensure proper types for merge keys - USE MAPPED SOURCE DATAFRAME
        for col in key_source:
            if col in df_source_mapped.columns:
                df_source_mapped[col] = df_source_mapped[col].astype(str)
                # Add extra trimming here - using the function defined at module level
                try:
                    # FIX: Use a check for Dask Series
                    col_data = df_source_mapped[col]
                    if hasattr(col_data, 'map_partitions'):
                        df_source_mapped[col] = col_data.map_partitions(
                            strip_strings, meta=('x', 'object')
                        )
                    else:
                        df_source_mapped[col] = strip_strings(col_data)
                    logger.debug(f"Applied extra trimming to source key column '{col}'")
                except Exception as e:
                    logger.warning(f"Could not apply extra trimming to source '{col}': {e}")
                    
        for col in key_target:
            df_target[col] = df_target[col].astype(str)
            # Add extra trimming here - using the function defined at module level
            try:
                # FIX: Use a check for Dask Series
                col_data = df_target[col]
                if hasattr(col_data, 'map_partitions'):
                    df_target[col] = col_data.map_partitions(
                        strip_strings, meta=('x', 'object')
                    )
                else:
                    df_target[col] = strip_strings(col_data)
                logger.debug(f"Applied extra trimming to target key column '{col}'")
            except Exception as e:
                logger.warning(f"Could not apply extra trimming to target '{col}': {e}")
        
        # Get mapped columns
        mapped_columns = {}
        for col, rules in mapping_config.get('mappings', {}).items():
            if rules.get('destinations'):
                for dest in rules['destinations']:
                    mapped_columns[col] = dest
        
        # Combine keys and mapped columns for merging
        merge_on_source = key_source
        merge_on_target = key_target
        logger.debug(f"Merging on: Source: {merge_on_source}, Target: {merge_on_target}")
        
        # Output debug info about dataframes before merge
        logger.debug(f"Columns in df_source_mapped: {df_source_mapped.columns.tolist()}")
        logger.debug(f"Columns in df_target: {df_target.columns.tolist()}")
        
        # Perform merge - USE MAPPED SOURCE DATAFRAME
        ddf_merged = dd.merge(
            df_source_mapped,  # Use the mapped dataframe, not the original
            df_target,
            left_on=merge_on_source,
            right_on=merge_on_target,
            how='outer',
            indicator=True
        )
        
        # Compute statistics using safe methods for Dask Series
        # Using .eq() method instead of == for robust Dask comparison
        both_count = ddf_merged['_merge'].eq('both').sum().compute()
        right_only_count = ddf_merged['_merge'].eq('right_only').sum().compute()
        left_only_count = ddf_merged['_merge'].eq('left_only').sum().compute()
        
        stats = {
            'total_match': both_count,
            'missing_source': right_only_count,
            'missing_target': left_only_count
        }
        
        return ddf_merged, stats
        
    except Exception as e:
        logger.error(f"Error in execute_matching_dask: {e}", exc_info=True)
        raise

def handle_large_file(file_path: str, chunk_size: int = 10000) -> pd.DataFrame:
    """Process large files in chunks to avoid memory issues."""
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

def generate_soda_yaml(validation_config: dict, table_name: str, business_rules: list = None) -> str:
    """
    Generate comprehensive SODA YAML configuration from validation and business rules.
    This improved version includes all validation types and detailed business rule checks.
    """
    sections = []
    
    # Set table name at top level
    sections.append(f"table: {table_name}")
    sections.append("checks:")
    
    # First section: Standard validation rules with clear organization
    if validation_config:
        validation_sections = []
        validation_sections.append("  # Column-level validation rules")
        
        for column, rules in validation_config.items():
            # Group checks by column for better organization
            column_checks = []
            
            # NOT NULL checks
            if rules.get(VRule.VALIDATE_NULLS.value, False):
                column_checks.append(f"  - not_null: {column}")
            
            # UNIQUE checks
            if rules.get(VRule.VALIDATE_UNIQUENESS.value, False):
                column_checks.append(f"  - unique: {column}")
            
            # MIN/MAX VALUE checks - with improved formatting
            if rules.get(VRule.VALIDATE_RANGE.value, False):
                min_val = rules.get(VRule.MIN_VALUE.value)
                max_val = rules.get(VRule.MAX_VALUE.value)
                
                if min_val is not None:
                    column_checks.append(f"  - min:")
                    column_checks.append(f"      column: {column}")
                    column_checks.append(f"      min_value: {min_val}")
                
                if max_val is not None:
                    column_checks.append(f"  - max:")
                    column_checks.append(f"      column: {column}")
                    column_checks.append(f"      max_value: {max_val}")
            
            # VALUES IN LIST checks - with proper value formatting
            if VRule.VALIDATE_LIST_OF_VALUES.value in rules:
                values = rules[VRule.VALIDATE_LIST_OF_VALUES.value]
                if values:
                    # Format values properly with quotes for strings
                    values_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in values])
                    column_checks.append(f"  - values_in:")
                    column_checks.append(f"      column: {column}")
                    column_checks.append(f"      values: [{values_str}]")
            
            # REGEX MATCH checks
            if VRule.VALIDATE_REGEX.value in rules and rules[VRule.VALIDATE_REGEX.value]:
                regex = rules[VRule.VALIDATE_REGEX.value]
                # Escape single quotes in regex pattern if needed
                safe_regex = regex.replace("'", "''")
                column_checks.append(f"  - regex_match:")
                column_checks.append(f"      column: {column}")
                column_checks.append(f"      regex: '{safe_regex}'")
            
            # Add all column checks to the validation sections
            if column_checks:
                validation_sections.append(f"  # Checks for column: {column}")
                validation_sections.extend(column_checks)
        
        # Add all validation sections if we have any
        if len(validation_sections) > 1:  # More than just the header
            sections.extend(validation_sections)
    
    # Second section: Business rules with improved SQL expressions
    if business_rules:
        sections.append("")  # Empty line for separation
        sections.append("  # Business rule checks")
        
        for rule in business_rules:
            rule_name = rule['name'].lower().replace(' ', '_')
            
            # Convert IF conditions to SQL format with proper conjunction
            conditions_sql = []
            for cond in rule['conditions']:
                operator_map = {
                    'equals': '=',
                    'not_equals': '!=',
                    'greater_than': '>',
                    'less_than': '<',
                    'greater_equal': '>=',
                    'less_equal': '<=',
                    'contains': 'LIKE',
                    'is_null': 'IS NULL',
                    'is_not_null': 'IS NOT NULL'
                }
                
                op = operator_map.get(cond['operator'], '=');
                col_name = cond['column']
                value_type = cond.get('value_type', 'value').lower()
                value = cond['value']
                
                # Format the condition based on operator and value type
                if cond['operator'] in ['is_null', 'is_not_null']:
                    # No value needed for NULL checks
                    conditions_sql.append(f"{col_name} {op}")
                elif cond['operator'] == 'contains':
                    # Use LIKE with wildcards for contains
                    if value_type == 'column':
                        conditions_sql.append(f"{col_name} {op} '%' || {value} || '%'")
                    else:
                        conditions_sql.append(f"{col_name} {op} '%{value}%'")
                else:
                    # Standard comparison
                    if value_type == 'column':
                        conditions_sql.append(f"{col_name} {op} {value}")
                    else:
                        # Try to determine if value is numeric or string
                        try:
                            float(value)  # Test if value can be converted to number
                            conditions_sql.append(f"{col_name} {op} {value}")
                        except (ValueError, TypeError):
                            # Use quotes for string values
                            conditions_sql.append(f"{col_name} {op} '{value}'")
            
            # Join all conditions with AND
            condition_str = " AND ".join(conditions_sql) if conditions_sql else "TRUE"
            
            # Convert THEN conditions to SQL assertions
            then_conditions_sql = []
            for then_cond in rule['then']:
                op = operator_map.get(then_cond['operator'], '=');
                col_name = then_cond['column']
                value_type = then_cond.get('value_type', 'value').lower()
                value = then_cond['value']
                
                # Format the THEN condition similar to IF conditions
                if then_cond['operator'] in ['is_null', 'is_not_null']:
                    then_conditions_sql.append(f"{col_name} {op}")
                elif then_cond['operator'] == 'contains':
                    if value_type == 'column':
                        then_conditions_sql.append(f"{col_name} {op} '%' || {value} || '%'")
                    else:
                        then_conditions_sql.append(f"{col_name} {op} '%{value}%'")
                else:
                    if value_type == 'column':
                        then_conditions_sql.append(f"{col_name} {op} {value}")
                    else:
                        try:
                            float(value)
                            then_conditions_sql.append(f"{col_name} {op} {value}")
                        except (ValueError, TypeError):
                            then_conditions_sql.append(f"{col_name} {op} '{value}'")
                            
            # Join all THEN conditions with AND
            then_condition_str = " AND ".join(then_conditions_sql) if then_conditions_sql else "TRUE"
                
            # Create the full SQL expression check
            sections.append(f"  - custom_sql_expr: {rule_name}")
            sections.append(f"    expression: >-")
            sections.append(f"      CASE")
            sections.append(f"        WHEN {condition_str}")
            sections.append(f"        THEN {then_condition_str}")
            sections.append(f"        ELSE TRUE")
            sections.append(f"      END")
    
    # Add row count check as a basic data quality check
    sections.append("")
    sections.append("  # Basic data quality checks")
    sections.append("  - row_count:")
    sections.append("      minimum: 1")
    
    # Add freshness check if appropriate
    sections.append("")
    sections.append("  # Add more checks as needed:")
    sections.append("  # - freshness:")
    sections.append("  #     column: updated_at")
    sections.append("  #     method: hours")
    sections.append("  #     maximum_age: 24")
    
    # Include YAML header with metadata
    header = [
        "# SODA Data Quality Checks",
        "# Generated automatically from validation rules and business rules",
        "# https://docs.soda.io/soda-cl/metrics-and-checks.html",
        ""
    ]
    
    # Combine all sections into YAML
    yaml_content = "\n".join(header + sections)
    
    return yaml_content

def validate_business_rule(rule, df):
    """Validates a single business rule against the dataframe."""
    try:
        mask = pd.Series(True, index=df.index)
        
        # Apply all conditions
        for condition in rule['conditions']:
            col = condition['column']
            op = condition['operator']
            val = condition['value']
            # Convert value to numeric if possible
            try:
                numeric_val = float(val)
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                is_numeric = True
            except ValueError:
                numeric_val = None
                numeric_col = None
                is_numeric = False
            if op == 'equals':
                mask &= df[col].astype(str) == str(val)
            elif op == 'not_equals':
                mask &= df[col].astype(str) != str(val)
            elif op == 'contains':
                mask &= df[col].astype(str).str.contains(str(val), na=False)
            elif op == 'is_null':
                mask &= df[col].isna()
            elif op == 'is_not_null':
                mask &= df[col].notna()
            elif is_numeric:
                if op == 'greater_than':
                    mask &= numeric_col > numeric_val
                elif op == 'less_than':
                    mask &= numeric_col < numeric_val
                elif op == 'greater_equal':
                    mask &= numeric_col >= numeric_val
                elif op == 'less_equal':
                    mask &= numeric_col <= numeric_val
                    
        # Check 'then' conditions where mask is True
        violations = []
        for then_condition in rule['then']:
            col = then_condition['column']
            op = then_condition['operator']
            val = then_condition['value']
            # Convert value to numeric if possible for 'then' conditions
            try:
                numeric_val = float(val)
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                is_numeric = True
            except ValueError:
                numeric_val = None
                numeric_col = None
                is_numeric = False
            # Apply validation only where conditions are met
            validation_mask = mask.copy()
            if op == 'equals':
                validation_mask &= df[col].astype(str) != str(val)
            elif op == 'not_equals':
                validation_mask &= df[col].astype(str) == str(val)
            elif op == 'contains':
                validation_mask &= ~df[col].astype(str).str.contains(str(val), na=False)
            elif op == 'is_null':
                validation_mask &= df[col].notna()
            elif op == 'is_not_null':
                validation_mask &= df[col].isna()
            elif is_numeric:
                if op == 'greater_than':
                    validation_mask &= numeric_col <= numeric_val
                elif op == 'less_than':
                    validation_mask &= numeric_col >= numeric_val
                elif op == 'greater_equal':
                    validation_mask &= numeric_col < numeric_val
                elif op == 'less_equal':
                    validation_mask &= numeric_col > numeric_val
            violations.extend(df[validation_mask].index.tolist())
            
        return list(set(violations))
    except Exception as e:
        logger.error(f"Error validating rule: {str(e)}")
        return f"Error validating rule: {str(e)}"

def validate_all_business_rules(df):
    """Validates all business rules and returns violations."""
    violations = {}
    for rule in st.session_state.business_rules:
        rule_violations = validate_business_rule(rule, df)
        if rule_violations:
            violations[rule['name']] = rule_violations
    return violations

def format_rule_as_sentence(rule: dict) -> str:
    """Format a business rule as a readable sentence."""
    try:
        # Format IF conditions
        if_parts = []
        for condition in rule['conditions']:
            operator_text = {
                'equals': 'is equal to',
                'not_equals': 'is not equal to',
                'greater_than': 'is greater than',
                'less_than': 'is less than',
                'greater_equal': 'is greater than or equal to',
                'less_equal': 'is less than or equal to',
                'contains': 'contains',
                'is_null': 'is empty',
                'is_not_null': 'is not empty'
            }.get(condition['operator'], condition['operator'])
            value_text = '' if condition['operator'] in ['is_null', 'is_not_null'] else f" {condition['value']}"
            if_parts.append(f"{condition['column']} {operator_text}{value_text}")
        if_clause = " AND ".join(if_parts)
        
        # Format THEN conditions
        then_parts = []
        for then_cond in rule['then']:
            operator_text = {
                'equals': 'must be equal to',
                'not_equals': 'must not be equal to',
                'greater_than': 'must be greater than',
                'less_than': 'must be less than',
                'greater_equal': 'must be greater than or equal to',
                'less_equal': 'must be less than or equal to',
                'contains': 'must contain',
                'is_null': 'must be empty',
                'is_not_null': 'must not be empty'
            }.get(then_cond['operator'], then_cond['operator'])
            value_text = '' if then_cond['operator'] in ['is_null', 'is_not_null'] else f" {then_cond['value']}"
            then_parts.append(f"{then_cond['column']} {operator_text}{value_text}")
        then_clause = " AND ".join(then_parts)
        
        # Combine into full sentence
        return f"IF {if_clause}, THEN {then_clause}"
    except Exception as e:
        logger.error(f"Error formatting rule: {str(e)}")
        return "Error formatting rule: {str(e)}"

def apply_string_transformations(df: pd.DataFrame, transformations: Dict[str, List[str]]) -> pd.DataFrame:
    """Apply string transformations to specified columns."""
    # Make a deep copy to ensure no references to original data
    df = df.copy(deep=True)
    
    for col, transforms in transformations.items():
        if col in df.columns:
            # Only apply to string/object columns
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                # Log original values for debugging
                sample_before = df[col].head(2).tolist()
                
                for transform in transforms:
                    if transform == 'trim':
                        # Enhanced trim that handles all Unicode whitespace
                        df[col] = df[col].astype(str).apply(
                            lambda x: re.sub(r'^\s+|\s+$', '', 
                                            unicodedata.normalize('NFKC', x)) if isinstance(x, str) else x
                        )
                        logger.info(f"Applied enhanced 'trim' to column '{col}'")
                    elif transform == 'upper':
                        df[col] = df[col].astype(str).str.upper()
                        logger.info(f"Applied 'upper' to column '{col}'")
                    elif transform == 'lower':
                        df[col] = df[col].astype(str).str.lower()
                        logger.info(f"Applied 'lower' to column '{col}'")
                        
                # Log transformed values for debugging
                sample_after = df[col].head(2).tolist()
                logger.debug(f"Column '{col}' before transforms: {sample_before}")
                logger.debug(f"Column '{col}' after transforms: {sample_after}")
                
                # Check if there are still whitespace issues after transformation
                if 'trim' in transforms:
                    # Build a pattern that matches a wider range of whitespace
                    whitespace_pattern = r'^\s+|\s+$'
                    still_has_whitespace = df[col].astype(str).str.match(whitespace_pattern).any()
                    if still_has_whitespace:
                        # Try one more clean using regex directly
                        df[col] = df[col].astype(str).apply(
                            lambda x: re.sub(whitespace_pattern, '', x) if isinstance(x, str) else x
                        )
                        logger.warning(f"Applied additional regex whitespace clean to column '{col}'")
    
    return df

@st.cache_data(ttl=600)
def get_dataframe_sample(df, sample_size=100):
    """Get a sample from a DataFrame with caching, handling both pandas and Dask"""
    if df is None:
        return pd.DataFrame()
    
    try:
        # Improved Dask DataFrame detection - use hasattr instead of type checking
        is_dask = hasattr(df, 'compute') and hasattr(df, 'npartitions')
        
        if is_dask:
            try:
                # Get a sample from the Dask DataFrame and compute it
                sample_df = df.head(sample_size)
                return sample_df.compute()
            except Exception as e:
                logger.error(f"Error sampling Dask DataFrame: {str(e)}")
                # Fallback: return sample without computing
                return sample_df
        else:
            # For pandas DataFrame
            try:
                actual_sample_size = min(sample_size, len(df))
                return df.head(actual_sample_size)
            except Exception as e:
                logger.error(f"Error getting pandas DataFrame sample: {str(e)}")
                # Return empty DataFrame with same columns if possible
                return pd.DataFrame(columns=getattr(df, 'columns', []))
    except Exception as e:
        logger.error(f"Error getting DataFrame sample: {str(e)}")
        # Return an empty DataFrame as a last resort
        if hasattr(df, 'columns'):
            return pd.DataFrame(columns(df.columns))
        else:
            return pd.DataFrame()

# Progressive DataFrame processing - better for large datasets
def process_dataframe_progressively(df, process_function, batch_size=10000):
    """Process a large dataframe in batches to avoid memory issues."""
    if df is None:
        return pd.DataFrame()
        
    # For Dask DataFrames
    if isinstance(df, dd.DataFrame):
        try:
            with st.spinner("Processing large dataset with Dask..."):
                # Let Dask handle the batching
                result = process_function(df)
                # Compute only what's needed
                if isinstance(result, dd.DataFrame):
                    # For display, get a sample
                    return result.head(100).compute()
                else:
                    return result
        except Exception as e:
            logger.error(f"Error processing Dask DataFrame: {str(e)}")
            return pd.DataFrame()
            
    # For Pandas DataFrames
    total_rows = len(df)
    if total_rows <= batch_size:
        return process_function(df)
        
    # Process in batches with progress bar
    results = []
    num_batches = (total_rows + batch_size - 1) // batch_size
    
    progress_bar = st.progress(0)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, total_rows)
        batch = df.iloc[start_idx:end_idx]
        
        # Process this batch
        batch_result = process_function(batch)
        results.append(batch_result)
        
        # Update progress
        progress = (i + 1) / num_batches
        progress_bar.progress(progress, f"Processing batch {i+1}/{num_batches} ({int(progress*100)}%)")
    
    # Combine results
    try:
        if isinstance(results[0], pd.DataFrame):
            return pd.concat(results)
        else:
            return results
    except Exception as e:
        logger.error(f"Error combining batch results: {str(e)}")
        return results[0] if results else pd.DataFrame()

# Add memory monitoring functions
import psutil
import os

def get_memory_usage():
    """Get current memory usage of the process"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Return memory usage in MB
    return {
        'rss': memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
        'vms': memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
        'percent': process.memory_percent()
    }

def log_memory_usage(message=""):
    """Log memory usage with an optional message"""
    memory = get_memory_usage()
    logger.info(f"Memory usage{' '+message if message else ''}: "
               f"{memory['rss']:.2f}MB RSS, {memory['percent']:.2f}% of total")
    return memory

# Add this function to optimize large DataFrames by converting to categorical where appropriate
def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage by converting appropriate columns to categorical"""
    df_optimized = df.copy()
    
    # Count unique values in object columns
    for col in df_optimized.select_dtypes(include=['object']).columns:
        num_unique = df_optimized[col].nunique()
        num_total = len(df_optimized[col])
        
        # If column has low cardinality (less than 50% unique values and at least 10 values),
        # convert to categorical
        if num_unique / num_total < 0.5 and num_unique > 10:
            df_optimized[col] = df_optimized[col].astype('category')
            logger.debug(f"Converted column '{col}' to category type")
    
    # For integer columns that have low cardinality, also convert to categorical
    for col in df_optimized.select_dtypes(include=['int64']).columns:
        num_unique = df_optimized[col].nunique()
        if num_unique < 100:  # Low cardinality for integer column
            df_optimized[col] = df_optimized[col].astype('category')
            logger.debug(f"Converted integer column '{col}' to category type")
    
    # Calculate memory savings
    orig_mem = df.memory_usage(deep=True).sum()
    new_mem = df_optimized.memory_usage(deep=True).sum()
    savings_pct = 100 * (1 - new_mem / orig_mem)
    
    logger.info(f"Memory optimization: {orig_mem/1024**2:.2f}MB → {new_mem/1024**2:.2f}MB "
               f"({savings_pct:.1f}% reduction)")
    
    return df_optimized

# Add function to track Dask memory usage
def log_dask_memory_usage():
    """Log memory usage specifically from Dask operations"""
    try:
        import dask
        if hasattr(dask, 'sizeof'):
            # Get size of cached dask arrays/dataframes
            from dask.utils import format_bytes
            cache_size = dask.sizeof.sizeof(dask.get_collection_cache())
            logger.info(f"Dask cache size: {format_bytes(cache_size)}")
            
        # Log Dask diagnostics if available
        if hasattr(dask, 'diagnostics'):
            stats = dask.diagnostics.get_task_stats()
            if stats:
                logger.info(f"Dask task stats: {stats}")
    except Exception as e:
        logger.warning(f"Could not log Dask memory usage: {e}")