import logging
import logging.config
from typing import Optional, Tuple, Dict, List, Union, Any
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
    def _should_use_dask(file_size: int) -> bool:
        # Use Dask if file size is greater than 500MB
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

    @staticmethod
    @st.cache_data(ttl=3600)
    def load_file(uploaded_file) -> Union[pd.DataFrame, dd.DataFrame]:
        try:
            max_size = STREAMLIT_CONFIG["max_file_size"] * 1024 * 1024
            if uploaded_file.size > max_size:
                raise ValueError(f"File size exceeds maximum limit of {STREAMLIT_CONFIG['max_file_size']}MB")

            file_path = Path(uploaded_file.name)
            if file_path.suffix not in ['.xlsx', '.csv']:
                raise ValueError("Unsupported file format. Only .xlsx and .csv files are supported.")

            use_dask = FileLoader._should_use_dask(uploaded_file.size)
            
            if file_path.suffix == '.xlsx':
                if use_dask:
                    # For xlsx, we need to read with pandas first, then convert to dask
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                    df = FileLoader._process_dataframe(df)
                    return dd.from_pandas(df, npartitions=DASK_CONFIG["npartitions"])
                else:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
            else:  # CSV file
                # Detect delimiter
                delimiter = FileLoader._detect_delimiter(uploaded_file)
                uploaded_file.seek(0)  # Reset file position after delimiter detection
                
                if use_dask:
                    return dd.read_csv(
                        uploaded_file,
                        sep=delimiter,
                        encoding='utf-8',
                        encoding_errors='replace',
                        on_bad_lines='warn'
                    )
                else:
                    try:
                        df = pd.read_csv(
                            uploaded_file,
                            sep=delimiter,
                            encoding='utf-8',
                            encoding_errors='replace',
                            engine='c',
                            on_bad_lines='warn',
                            low_memory=False
                        )
                    except Exception as e:
                        logger.warning(f"Failed to read with 'c' engine: {str(e)}. Trying 'python' engine...")
                        uploaded_file.seek(0)  # Reset file position
                        df = pd.read_csv(
                            uploaded_file,
                            sep=delimiter,
                            encoding='utf-8',
                            encoding_errors='replace',
                            engine='python',
                            on_bad_lines='skip'
                        )
            
            return FileLoader._process_dataframe(df)

        except Exception as e:
            logger.error("Error loading file", exc_info=True, extra={
                'filename': uploaded_file.name,
                'file_size': uploaded_file.size,
                'content_type': uploaded_file.type
            })
            raise RuntimeError(f"Failed to load file: {str(e)}") from e

    @staticmethod
    def _process_dataframe(df: Union[pd.DataFrame, dd.DataFrame]) -> Union[pd.DataFrame, dd.DataFrame]:
        try:
            if isinstance(df, dd.DataFrame):
                # Process Dask DataFrame
                df = df.map_partitions(lambda pdf: pdf.convert_dtypes())
                df.columns = [col.strip().replace("\ufeff", "") for col in df.columns]
                return df
            else:
                # Process Pandas DataFrame
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
    df = df.copy()
    key_source = mapping_config.get('key_source', [])
    
    # Collect all aggregations to perform them together
    agg_cols = {}
    
    # First handle conversions and prepare aggregations
    for col, rules in mapping_config.get('mappings', {}).items():
        if rules["function"] == Functions.CONVERSION.value:
            if rules.get("transformation"):
                mapping_dict = json.loads(rules["transformation"])
                if is_dask:
                    # For Dask DataFrame, use map_partitions with explicit function
                    df[col] = df[col].map_partitions(_convert_using_dict, mapping_dict)
                else:
                    df[col] = _convert_using_dict(df[col], mapping_dict)
                agg_cols[col] = 'first'
        elif rules["function"] == Functions.DIRECT.value:
            agg_cols[col] = 'first'
        elif rules["function"] == Functions.AGGREGATION.value:
            agg_cols[col] = rules["transformation"]
    
    # If there are any columns to aggregate
    if agg_cols:
        try:
            # Keep only necessary columns
            columns_to_keep = key_source + list(agg_cols.keys())
            df = df[columns_to_keep].copy()
            
            if is_dask:
                # For Dask DataFrame, compute aggregations
                grouped = df.groupby(key_source)
                results = []
                
                # Process each column's aggregation separately
                for col, agg_func in agg_cols.items():
                    if agg_func == 'first':
                        # Special handling for 'first' aggregation
                        agg_result = grouped[col].first().reset_index()
                    else:
                        agg_result = grouped[col].agg(agg_func).reset_index()
                    results.append(agg_result)
                
                # Merge results if we have multiple columns
                if results:
                    df = results[0]
                    for other_df in results[1:]:
                        df = df.merge(other_df, on=key_source)
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

@st.cache_data
def execute_matching_dask(df_source: Union[pd.DataFrame, dd.DataFrame], 
                         df_target: Union[pd.DataFrame, dd.DataFrame], 
                         mapping_config: dict) -> Tuple[dd.DataFrame, dict]:
    """Execute the matching process using Dask."""
    try:
        # Convert to Dask DataFrame if input is Pandas DataFrame
        if isinstance(df_source, pd.DataFrame):
            df_source = dd.from_pandas(df_source, npartitions=DASK_CONFIG["npartitions"])
        if isinstance(df_target, pd.DataFrame):
            df_target = dd.from_pandas(df_target, npartitions=DASK_CONFIG["npartitions"])
            
        # Get keys and prepare for merge
        key_source = mapping_config.get('key_source', [])
        key_target = mapping_config.get('key_target', [])
        
        if not key_source or not key_target:
            raise ValueError("Source and target keys must be defined")
        
        # Apply mapping rules to source DataFrame
        df_source = apply_rules(df_source, mapping_config)
        
        # Ensure proper types for merge keys
        for col in key_source:
            df_source[col] = df_source[col].astype(str)
        for col in key_target:
            df_target[col] = df_target[col].astype(str)
        
        # Get mapped columns
        mapped_columns = {}
        for col, rules in mapping_config.get('mappings', {}).items():
            if rules.get('destinations'):
                for dest in rules['destinations']:
                    mapped_columns[col] = dest
        
        # Combine keys and mapped columns for merging
        merge_on_source = key_source + list(mapped_columns.keys())
        merge_on_target = key_target + list(mapped_columns.values())
        
        # Perform merge
        ddf_merged = dd.merge(
            df_source,
            df_target,
            left_on=merge_on_source,
            right_on=merge_on_target,
            how='outer',
            indicator=True
        )
        
        # Compute statistics (using Dask operations)
        both_mask = ddf_merged['_merge'] == 'both'
        right_mask = ddf_merged['_merge'] == 'right_only'
        left_mask = ddf_merged['_merge'] == 'left_only'
        
        stats = {
            'total_match': both_mask.sum().compute(),
            'missing_source': right_mask.sum().compute(),
            'missing_target': left_mask.sum().compute()
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
