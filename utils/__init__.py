# Import file loading utilities
from .file_loading import FileLoader

import streamlit as st

# Import data validation utilities
from .data_validation import DataValidator, validate_all_business_rules

# Import transformation utilities
from .transformation import format_rule_as_sentence, execute_matching_dask, generate_soda_yaml

# Import report utilities
from .report_utils import (
    generate_pdf_report,
    generate_excel_report,
    generate_json_report,
    open_pdf,
    prepare_report_data
)

# Import export utilities
from .export_utils import json_serialize, generate_excel_report

# Import mapping helpers if they exist
try:
    from .mapping_helpers import smart_column_mapper, check_key_uniqueness
except ImportError:
    pass

try:
    from .mapping_utils import load_mapping_from_file, save_mapping_to_json, apply_mapping_to_session
except ImportError:
    pass

# Import data type utilities
from .data_type_utils import (
    get_descriptive_type, 
    is_numeric_string, 
    looks_like_date,
    create_mini_bar,
    create_null_visualization_card,
    create_column_grouping_ui  # Add this missing import
)

import pandas as pd
from typing import Union, Dict, Any, Optional

def clean_dataframe_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a DataFrame for display, handling complex objects and ensuring Arrow compatibility.
    
    This function prepares DataFrames for display in Streamlit by ensuring all data
    is compatible with Arrow serialization, which is required by Streamlit's st.dataframe.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        Cleaned DataFrame ready for display with Arrow-compatible data types
    """
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Make a copy to avoid modifying the original
    display_df = df.copy()
    
    # Process each column to ensure Arrow compatibility
    for col in display_df.columns:
        # Handle different data types
        col_dtype = display_df[col].dtype
        
        # 1. Handle categorical columns
        if pd.api.types.is_categorical_dtype(display_df[col]):
            # Convert categorical to string to avoid "Cannot setitem on a Categorical with a new category" error
            display_df[col] = display_df[col].astype(str)
        
        # 2. Handle object columns - convert to string for display and try to detect dates
        elif col_dtype.name == 'object':
            # Check for date patterns in column name
            is_likely_date = any(keyword in col.lower() for keyword in 
                               ['date', 'time', 'dt', 'timestamp', 'year', 'month', 'day'])
                               
            # Sample non-null values to check if they might be dates
            mask = display_df[col].notna()
            if mask.any():
                sample_values = display_df.loc[mask, col].head(10).tolist()
                sample_value = display_df.loc[mask, col].iloc[0]
                
                # Try to convert date-like columns to datetime
                if is_likely_date or _looks_like_date(sample_values):
                    try:
                        display_df[col] = pd.to_datetime(display_df[col], errors='coerce')
                        continue
                    except:
                        # Fallback to string if conversion fails
                        pass
                
                # Handle complex types that Arrow can't serialize
                if not isinstance(sample_value, (str, int, float, bool)):
                    display_df[col] = display_df[col].apply(
                        lambda x: str(x) if x is not None else None
                    )
                # Convert all other object columns to string for consistent display
                else:
                    display_df[col] = display_df[col].astype(str)
        
        # 3. Handle datetime columns to ensure consistent formatting
        elif pd.api.types.is_datetime64_dtype(display_df[col]):
            # Keep datetime type as Arrow handles it well
            pass
        
        # 4. Handle any problematic numeric columns
        elif pd.api.types.is_numeric_dtype(display_df[col]):
            # Check if column contains inf or -inf which can cause Arrow issues
            if (display_df[col] == float('inf')).any() or (display_df[col] == float('-inf')).any():
                # Replace inf values with None
                display_df[col] = display_df[col].replace([float('inf'), float('-inf')], None)
                
        # 5. Explicitly convert any unknown types to strings as a last resort
        elif col_dtype.name not in ('int64', 'float64', 'bool', 'datetime64[ns]', 'timedelta64[ns]'):
            try:
                display_df[col] = display_df[col].astype(str)
            except Exception:
                # If conversion fails, replace with None values
                display_df[col] = None
    
    return display_df

def _looks_like_date(values: list) -> bool:
    """
    Check if a list of values looks like dates.
    
    Args:
        values: List of values to check
        
    Returns:
        True if values appear to be dates
    """
    # Common date formats to check
    date_patterns = [
        # Check for slashes (MM/DD/YYYY or DD/MM/YYYY)
        lambda x: isinstance(x, str) and len(x) >= 8 and len(x) <= 10 and x.count('/') == 2,
        # Check for dashes (YYYY-MM-DD)
        lambda x: isinstance(x, str) and len(x) == 10 and x[4:5] == '-' and x[7:8] == '-',
        # Check for dots (DD.MM.YYYY)
        lambda x: isinstance(x, str) and len(x) >= 8 and len(x) <= 10 and x.count('.') == 2,
        # Check for timestamps (contains both date and time parts)
        lambda x: isinstance(x, str) and len(x) >= 16 and 
                 (((':' in x) and ('-' in x or '/' in x)) or 
                  (('T' in x) and x.count(':') >= 1))
    ]
    
    if not values or len(values) < 2:
        return False
        
    # Check if at least 70% of values match any date pattern
    matches = 0
    for value in values:
        if value is None:
            continue
        if any(pattern(value) for pattern in date_patterns):
            matches += 1
            
    return (matches / len(values)) >= 0.7

def safe_load_file(file, file_type: str = None) -> pd.DataFrame:
    """
    Safely load a file into a DataFrame, handling errors and metadata properly.
    
    Args:
        file: File object from st.file_uploader
        file_type: Optional file type override
        
    Returns:
        DataFrame from the loaded file, or empty DataFrame if loading fails
    """
    from .file_loading import FileLoader
    
    try:
        # Call the FileLoader and unpack the tuple return value
        df, metadata, error = FileLoader.load_file(file, file_type)
        
        # If there's an error, display it
        if error:
            st.error(f"Error loading file: {error}")
            return pd.DataFrame()
            
        # Display loading info
        if metadata:
            st.success(f"File loaded successfully: {metadata.get('rows', 0)} rows, {metadata.get('columns', 0)} columns")
            
            # Show additional metadata in expandable section if it exists
            if len(metadata) > 2:  # More than just rows and columns
                with st.expander("File metadata"):
                    for key, value in metadata.items():
                        if key not in ['rows', 'columns']:
                            st.write(f"**{key}**: {value}")
        
        # Return only the DataFrame
        return df
        
    except Exception as e:
        st.error(f"Unexpected error loading file: {str(e)}")
        return pd.DataFrame()

# Make sure clean_dataframe_for_display is exported explicitly
# and overrides any imported function with the same name
__all__ = [
    'FileLoader',
    'clean_dataframe_for_display',  # Our unified function
    'DataValidator',
    'validate_all_business_rules',
    'format_rule_as_sentence',
    'generate_soda_yaml',
    'execute_matching_dask',
    'json_serialize',
    'generate_pdf_report',
    'generate_excel_report',
    'safe_load_file',  # Add new function to exports
    'smart_column_mapper',
    'check_key_uniqueness',
    'load_mapping_from_file',
    'save_mapping_to_json',
    'apply_mapping_to_session',
    # Add new utilities
    'get_descriptive_type',
    'is_numeric_string',
    'looks_like_date',
    'create_mini_bar',
    'create_null_visualization_card',
    'create_column_grouping_ui'  # Add this missing export
]

# Utilities package initialization
