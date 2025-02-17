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
            mapping_config = json.loads(content)
            
            # Update the entire mapping configuration
            st.session_state.mapping = {
                "mappings": mapping_config.get('mappings', {}),
                "key_source": mapping_config.get('key_source', []),
                "key_target": mapping_config.get('key_target', [])
            }
            # Also update individual keys for backwards compatibility
            st.session_state.key_source = mapping_config.get('key_source', [])
            st.session_state.key_target = mapping_config.get('key_target', [])
            
            # Debug information
            st.write("Loaded mapping configuration:")
            st.write({
                "mappings": len(mapping_config.get('mappings', {})),
                "key_source": mapping_config.get('key_source', []),
                "key_target": mapping_config.get('key_target', [])
            })
            
            st.success("Mapping rules loaded successfully.")
            st.session_state.update_ui = True
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            st.error(f"Error loading mapping configuration: {str(e)}")
            st.write("Error details:", str(e))

    @staticmethod
    def load_validations_from_json(json_file) -> None:
        """Load validation rules from JSON file."""
        try:
            content = json_file.read()
            loaded_rules = json.loads(content)
            
            # Convert the loaded rules into our validation format
            validation_rules = {}
            for col, rules in loaded_rules.items():
                validation_rules[col] = {}
                
                # Handle simple boolean validations
                if rules.get('validate_nulls'):
                    validation_rules[col][VRule.VALIDATE_NULLS.value] = True
                if rules.get('validate_uniqueness'):
                    validation_rules[col][VRule.VALIDATE_UNIQUENESS.value] = True
                
                # Handle list of values
                if rules.get('validate_list_of_values'):
                    values = rules.get('validate_list_of_values')
                    if isinstance(values, (list, tuple)):
                        validation_rules[col][VRule.VALIDATE_LIST_OF_VALUES.value] = values
                    elif isinstance(values, str):
                        validation_rules[col][VRule.VALIDATE_LIST_OF_VALUES.value] = [v.strip() for v in values.split(',')]
                
                # Handle regex
                if rules.get('validate_regex'):
                    validation_rules[col][VRule.VALIDATE_REGEX.value] = rules['validate_regex']
                
                # Handle range validation
                if rules.get('validate_range'):
                    validation_rules[col][VRule.VALIDATE_RANGE.value] = True
                    # Handle min value if present
                    if 'min_value' in rules:
                        validation_rules[col][VRule.MIN_VALUE.value] = rules['min_value']
                    # Handle max value if present
                    if 'max_value' in rules:
                        validation_rules[col][VRule.MAX_VALUE.value] = rules['max_value']
            
            st.session_state.validation_rules = validation_rules
            
            # Debug information
            st.write("Loaded validation rules:")
            st.write(validation_rules)
            
            st.success("Validation rules loaded successfully.")
            st.session_state.update_ui = True
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON format: {str(e)}")
            st.write("Invalid JSON content:", content)
        except Exception as e:
            st.error(f"Error loading validation rules: {str(e)}")
            st.write("Error details:", str(e))
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
