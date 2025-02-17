import logging
from typing import Optional, Dict, List, Any, Tuple
import streamlit as st
import pandas as pd
import plotly.express as px
import dask.dataframe as dd
import json
from utils import FileLoader, ConfigLoader
from state import SessionState
from config import DASK_CONFIG, STREAMLIT_CONFIG
from constants import FILE_TYPES, Column, ValidationRule as VRule

logger = logging.getLogger(__name__)

# Performance optimization: Use st.cache_data for expensive computations
@st.cache_data
def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute and cache dataframe statistics.
    
    Args:
        df: Input dataframe.
        
    Returns:
        Dictionary containing summary statistics, data types, and null counts.
    """
    return {
        "summary": df.describe(include='all'),
        "dtypes": df.dtypes,
        "null_counts": df.isnull().sum().reset_index()
    }

@st.cache_data
def execute_matching_dask(
    df_source: pd.DataFrame, 
    df_target: pd.DataFrame, 
    key_source: List[str], 
    key_target: List[str]
) -> Tuple[dd.DataFrame, Dict[str, int]]:
    """Execute matching operation with caching.
    
    Args:
        df_source: Source dataframe.
        df_target: Target dataframe.
        key_source: List of source key columns.
        key_target: List of target key columns.
        
    Returns:
        Tuple containing:
        - Dask DataFrame with merge results.
        - Dictionary with matching statistics.
    """
    npartitions = DASK_CONFIG["default_npartitions"]
    ddf_source = dd.from_pandas(df_source, npartitions=npartitions)
    ddf_target = dd.from_pandas(df_target, npartitions=npartitions)

    ddf_final = ddf_source.merge(
        ddf_target, 
        left_on=key_source, 
        right_on=key_target, 
        how="outer", 
        indicator=True
    )

    stats = {
        "total_match": int(ddf_final[ddf_final['_merge'] == 'both'].shape[0].compute()),
        "missing_source": int(ddf_final[ddf_final['_merge'] == 'right_only'].shape[0].compute()),
        "missing_target": int(ddf_final[ddf_final['_merge'] == 'left_only'].shape[0].compute())
    }

    return ddf_final, stats

# Initialize session state
SessionState.initialize()

def main():
    """Main function to handle the file mapping process."""
    st.title("File Mapping Process")
    
    if st.session_state.step == 1:
        handle_source_file_upload()
    elif st.session_state.step == 2:
        handle_target_file_upload()
    elif st.session_state.step == 3:
        handle_mapping_rules()
    elif st.session_state.step == 4:
        handle_matching_execution()
    elif st.session_state.step == 5:
        handle_validation_rules()
    elif st.session_state.step == 6:
        handle_data_validation()

def handle_source_file_upload():
    """Handle the source file upload step."""
    st.header("Step 1: Load Source File")
    uploaded_source = st.file_uploader("Load Source File", type=FILE_TYPES)
    if uploaded_source:
        df = FileLoader.load_file(uploaded_source)
        if df is not None:
            SessionState.set_dataframe('df_source', df)
            display_metadata(df, "Source Data")
            if st.button("Next Step"):
                st.session_state.step = 2
                st.rerun()

def handle_target_file_upload():
    """Handle the target file upload step."""
    st.header("Step 2: Load Target File")
    uploaded_target = st.file_uploader("Load Target File", type=FILE_TYPES)
    if uploaded_target:
        df = FileLoader.load_file(uploaded_target)
        if df is not None:
            SessionState.set_dataframe('df_target', df)
            display_metadata(df, "Target Data")
            if st.button("Next Step"):
                st.session_state.step = 3
                st.rerun()

def handle_mapping_rules():
    """Handle the mapping rules definition step."""
    st.header("Step 3: Define Matching Rules and Keys")
    
    with st.popover(":blue[Upload JSON File for Mapping]", icon=":material/publish:", use_container_width=True):
        uploaded_json = st.file_uploader("Load Mapping Rules from JSON", type=["json"])
        if uploaded_json:
            ConfigLoader.load_mapping_from_json(uploaded_json)
            if st.session_state.get("update_ui"):
                st.session_state.update_ui = False

    st.session_state.key_source = st.multiselect("Select the search key(s) in the source", st.session_state.df_source.columns, default=st.session_state.mapping.get("key_source", []))
    st.session_state.key_target = st.multiselect("Select the search key(s) in the target", st.session_state.df_target.columns, default=st.session_state.mapping.get("key_target", []))
    
    mapping_config = {"key_source": st.session_state.key_source, "key_target": st.session_state.key_target, "mappings": st.session_state.mapping.get("mappings", {})}
    
    for col in st.session_state.df_source.columns:
        if col not in st.session_state.key_source:
            with st.expander(f"Configure '{col}'"):
                option = st.radio(f"What do you want to do with '{col}'?", ["Ignore", "Map"], key=f"option_{col}", index=0 if col not in mapping_config["mappings"] else 1)
                if option == "Map":
                    mapped_cols = st.multiselect(f"Map '{col}' to:", list(st.session_state.df_target.columns), key=f"map_{col}", default=mapping_config["mappings"].get(col, {}).get("destinations", []))
                    function = st.selectbox(f"Mapping Type for '{col}'", ["Direct Match", "Aggregation", "Conversion"], key=f"func_{col}", index=["Direct Match", "Aggregation", "Conversion"].index(mapping_config["mappings"].get(col, {}).get("function", "Direct Match")))
                    transformation = mapping_config["mappings"].get(col, {}).get("transformation", None)
                    if function == "Aggregation":
                        transformation = st.selectbox("Aggregation Type", ["Sum", "Mean", "Median", "Max", "Min"], key=f"agg_{col}", index=["Sum", "Mean", "Median", "Max", "Min"].index(transformation) if transformation else 0)
                    elif function == "Conversion":
                        transformation = st.text_area("Define Conversion Dictionary (JSON)", transformation or "{}", key=f"conv_{col}")
                        try:
                            dict_data = json.loads(transformation)
                            dict_df = pd.DataFrame(list(dict_data.items()), columns=["Source", "Target"])
                            st.write("Preview of Conversion Dictionary:")
                            st.dataframe(dict_df)
                        except json.JSONDecodeError:
                            st.error("The conversion dictionary format is not valid. Make sure it is in correct JSON.")
                    mapping_config["mappings"][col] = {"destinations": mapped_cols, "function": function, "transformation": transformation}
    
    st.session_state.mapping = mapping_config
    
    if st.checkbox("Show Matching Configuration (JSON)", value=False, key="show_json"):
        st.json(st.session_state.mapping)

    if st.button("Next Step"):
        st.session_state.step = 4
        st.rerun()

def handle_matching_execution():
    """Handle the matching execution step."""
    st.header("Step 4: Execute Matching")
    if st.session_state.key_source and st.session_state.key_target:
        st.write("Matching process in progress...")
        ddf_final, stats = execute_matching_dask(
            st.session_state.df_source,
            st.session_state.df_target,
            st.session_state.key_source,
            st.session_state.key_target
        )
        
        # Display statistics
        st.write("### Matching Summary")
        st.write(f"Total matching records: {stats['total_match']}")
        st.write(f"Total missing in source: {stats['missing_source']}")
        st.write(f"Total missing in target: {stats['missing_target']}")

        # Store matching results
        st.session_state.matching_results = ddf_final[ddf_final['_merge'] == 'both'].compute()

        # Display sample results
        sample_size = min(DASK_CONFIG["sample_size"], len(ddf_final))
        df_final_sample = ddf_final.compute().sample(n=sample_size, random_state=42)
        
        st.write("### Matching Records (Sample)")
        st.dataframe(df_final_sample[df_final_sample['_merge'] == 'both'])
        
        st.write("### Non-Matching Records")
        non_matching_sample = df_final_sample[df_final_sample['_merge'] != 'both']
        st.write("Sample:")
        st.dataframe(non_matching_sample)
        
        # Add download button for full non-matching records
        if st.button("Generate Full Non-Matching Records"):
            with st.spinner("Generating full dataset..."):
                # Compute full non-matching records
                full_non_matching = ddf_final[ddf_final['_merge'] != 'both'].compute()
                
                # Convert to CSV
                csv = full_non_matching.to_csv(index=False)
                
                # Create download button
                st.download_button(
                    label="Download Full Non-Matching Records",
                    data=csv,
                    file_name="non_matching_records.csv",
                    mime="text/csv"
                )
                
                # Show total records
                st.write(f"Total non-matching records: {len(full_non_matching)}")
    else:
        st.warning("Define a valid search key before executing the check.")

    if st.button("Step back"):
        st.session_state.step = 3
        st.rerun()
    
    if st.button("Configure data validations"):
        st.session_state.step = 5
        st.rerun()

def handle_validation_rules():
    """Handle the validation rules definition step."""
    st.header("Step 5: Mapping Validation Rules")

    business_rules_config = {}

    st.subheader("Direct Validations")
    with st.popover(":blue[Upload JSON File]", icon=":material/publish:", use_container_width=True):
        uploaded_json = st.file_uploader("Load Validation Rules from JSON", type=["json"])
        if uploaded_json:
            ConfigLoader.load_validations_from_json(uploaded_json)
            if st.session_state.get("update_ui"):
                st.session_state.update_ui = False

    validation_rules = {}
    for col in st.session_state.df_target.columns:
        with st.popover(f"Configure validation for '{col}'", use_container_width=True):
            # Initialize column rules dictionary if it doesn't exist
            validation_rules[col] = validation_rules.get(col, {})
            
            validate_nulls = st.checkbox("Check Nulls", key=f"nulls_{col}", 
                value=st.session_state.validation_rules.get(col, {}).get(VRule.VALIDATE_NULLS.value, False))
            if validate_nulls:
                validation_rules[col][VRule.VALIDATE_NULLS.value] = True
            
            validate_unique = st.checkbox("Check Uniqueness", key=f"unique_{col}", 
                value=st.session_state.validation_rules.get(col, {}).get(VRule.VALIDATE_UNIQUENESS.value, False))
            if validate_unique:
                validation_rules[col][VRule.VALIDATE_UNIQUENESS.value] = True
            
            validate_domain = st.checkbox("Check List of Values (comma separated)", key=f"domain_{col}", 
                value=st.session_state.validation_rules.get(col, {}).get(VRule.VALIDATE_LIST_OF_VALUES.value, False))
            if validate_domain:
                domain_values = st.text_input(
                    "Allowed Values", 
                    key=f"domain_values_{col}",
                    value=",".join(st.session_state.validation_rules.get(col, {}).get(VRule.VALIDATE_LIST_OF_VALUES.value, []))
                )
                validation_rules[col][VRule.VALIDATE_LIST_OF_VALUES.value] = domain_values.split(',')
            
            validate_regex = st.checkbox("Check Format (Regex)", key=f"regex_{col}", 
                value=st.session_state.validation_rules.get(col, {}).get(VRule.VALIDATE_REGEX.value, False))
            if validate_regex:
                regex_pattern = st.text_input(
                    "Regular Expression",
                    key=f"regex_pattern_{col}",
                    value=st.session_state.validation_rules.get(col, {}).get(VRule.VALIDATE_REGEX.value, "")
                )
                validation_rules[col][VRule.VALIDATE_REGEX.value] = regex_pattern
            
            if pd.api.types.is_numeric_dtype(st.session_state.df_target[col]):
                validate_range = st.checkbox("Check Range Value", key=f"range_{col}", 
                    value=st.session_state.validation_rules.get(col, {}).get(VRule.VALIDATE_RANGE.value, False))
                if validate_range:
                    validation_rules[col][VRule.VALIDATE_RANGE.value] = True
                    
                    current_rules = st.session_state.validation_rules.get(col, {})
                    has_min = st.checkbox("Set minimum value", key=f"has_min_{col}",
                        value=current_rules.get(VRule.MIN_VALUE.value) is not None)
                    if has_min:
                        stored_min = current_rules.get(VRule.MIN_VALUE.value, 0.0)
                        min_value = st.number_input(
                            "Minimum Value", 
                            value=float(stored_min if stored_min is not None else 0.0),
                            key=f"min_{col}"
                        )
                        validation_rules[col][VRule.MIN_VALUE.value] = min_value
                    else:
                        validation_rules[col][VRule.MIN_VALUE.value] = None
                    
                    has_max = st.checkbox("Set maximum value", key=f"has_max_{col}",
                        value=current_rules.get(VRule.MAX_VALUE.value) is not None)
                    if has_max:
                        stored_max = current_rules.get(VRule.MAX_VALUE.value, 0.0)
                        max_value = st.number_input(
                            "Maximum Value", 
                            value=float(stored_max if stored_max is not None else 0.0),
                            key=f"max_{col}"
                        )
                        validation_rules[col][VRule.MAX_VALUE.value] = max_value
                    else:
                        validation_rules[col][VRule.MAX_VALUE.value] = None

            # Remove empty validation rules
            if not any(validation_rules[col].values()):
                validation_rules.pop(col, None)

    st.session_state.validation_rules = validation_rules

    if st.checkbox("Show Validation Configuration (JSON)", value=False, key="show_json"):
        st.json(st.session_state.validation_rules)

    if st.button("Next Step"):
        st.session_state.step = 6
        st.rerun()

def handle_data_validation():
    """Handle the data validation step."""
    st.header("Step 6: Data Validation")
    if st.session_state.df_target is not None and st.session_state.validation_rules:
        validation_results = execute_data_validation(st.session_state.df_target, st.session_state.validation_rules)
        st.session_state.validation_results = validation_results
    else:
        st.warning("Load the target data and define validation rules before executing the validation.")
    
    if st.button("Step back"):
        st.session_state.step = 5
        st.rerun()

def display_metadata(df: pd.DataFrame, title: str):
    """Display metadata of the dataframe.
    
    Args:
        df: Input dataframe.
        title: Title for the metadata section.
    """
    if df is not None:
        st.subheader(title)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        st.write("Statistical Summary:")
        st.write(df.describe(include='all'))
        
        st.write("Data Types:")
        st.write(df.dtypes)
        
        st.write("Null Values by Column:")
        null_counts = df.isnull().sum().reset_index()
        null_counts.columns = [Column.NAME, Column.NULL_VALUES]
        fig_nulls = px.bar(null_counts, x=Column.NAME, y=Column.NULL_VALUES, title='Null Values by Column')
        st.plotly_chart(fig_nulls)

def execute_data_validation(df: pd.DataFrame, validation_rules: dict) -> list:
    """Execute data validation based on defined rules.
    
    Args:
        df: Input dataframe.
        validation_rules: Dictionary of validation rules.
        
    Returns:
        List of validation results.
    """
    st.write("Executing data validation...")

    validation_results = []

    for col, rules in validation_rules.items():
        total_records = len(df)
        if VRule.VALIDATE_NULLS.value in rules and rules[VRule.VALIDATE_NULLS.value]:
            nulls = df[col].isnull().sum()
            validation_results.append({
                Column.NAME: col,
                "Rule": "Null values",
                "Pass": total_records - nulls,
                "Fail": nulls,
                "Pass %": f"{((total_records - nulls) / total_records) * 100:.2f}%"
            })
        
        if VRule.VALIDATE_UNIQUENESS.value in rules and rules[VRule.VALIDATE_UNIQUENESS.value]:
            unique = df[col].nunique() == len(df)
            validation_results.append({
                Column.NAME: col,
                "Rule": "Unique values",
                "Pass": total_records if unique else 0,
                "Fail": 0 if unique else total_records - df[col].nunique(),
                "Pass %": f"{(total_records / total_records) * 100:.2f}%" if unique else "0.00%"
            })
        
        if VRule.VALIDATE_LIST_OF_VALUES.value in rules:
            allowed_values = rules[VRule.VALIDATE_LIST_OF_VALUES.value]
            domain = df[~df[col].isin(allowed_values)].shape[0]
            validation_results.append({
                Column.NAME: col,
                "Rule": "Values outside allowed list",
                "Pass": total_records - domain,
                "Fail": domain,
                "Pass %": f"{((total_records - domain) / total_records) * 100:.2f}%"
            })
        
        if VRule.VALIDATE_REGEX.value in rules:
            regex_pattern = rules[VRule.VALIDATE_REGEX.value]
            regex = df[~df[col].str.match(regex_pattern, na=False)].shape[0]
            validation_results.append({
                Column.NAME: col,
                "Rule": "Values not matching regex",
                "Pass": total_records - regex,
                "Fail": regex,
                "Pass %": f"{((total_records - regex) / total_records) * 100:.2f}%"
            })
        
        if VRule.VALIDATE_RANGE.value in rules:
            try:
                min_value = rules.get(VRule.MIN_VALUE.value)
                max_value = rules.get(VRule.MAX_VALUE.value)
                
                # Convert to numeric and handle NaN values
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                
                # Build the condition based on available limits
                condition = numeric_col.notnull()  # Start with non-null values
                if min_value is not None:
                    condition &= (numeric_col >= float(min_value))
                if max_value is not None:
                    condition &= (numeric_col <= float(max_value))
                
                # Count values outside the range
                out_of_range = len(df) - condition.sum()
                
                validation_results.append({
                    Column.NAME: col,
                    "Rule": "Values out of range",
                    "Pass": total_records - out_of_range,
                    "Fail": out_of_range,
                    "Pass %": f"{((total_records - out_of_range) / total_records) * 100:.2f}%"
                })
            except (TypeError, ValueError) as e:
                st.warning(f"Could not validate range for column {col}: {str(e)}")
                continue

    display_validation_summary(validation_results)
    display_detailed_validation_results(df, validation_results, validation_rules)

    return validation_results

def display_validation_summary(validation_results: list) -> None:
    """Display a summary of validation results in a styled dataframe.
    
    Args:
        validation_results: List of validation results.
    """
    st.write("### Validation Results Summary")
    summary_df = pd.DataFrame(validation_results)
    
    def color_pass_percentage(val):
        """Color formatting for pass percentage values.
        
        Args:
            val: Pass percentage value.
            
        Returns:
            CSS style string.
        """
        if isinstance(val, str) and val.endswith('%'):
            percentage = float(val[:-1])
            return 'background-color: green' if percentage >= 50 else 'background-color: red'
        return ''
    
    st.dataframe(summary_df.style.applymap(color_pass_percentage))

def display_detailed_validation_results(df: pd.DataFrame, validation_results: list, validation_rules: dict) -> None:
    """Display detailed validation results for each column with failures.
    
    Args:
        df: Input dataframe.
        validation_results: List of validation results.
        validation_rules: Dictionary of validation rules.
    """
    st.write("### Detailed Validation Results")
    
    for col_results in validation_results:
        col = col_results[Column.NAME]
        if col_results["Fail"] > 0:
            st.subheader(f"Column: {col}")
            
            # Display specific failure details based on rule type
            rule = col_results["Rule"]
            if rule == "Null values":
                st.write(f"Null values: {col_results['Fail']}")
                st.dataframe(df[df[col].isnull()])
            
            elif rule == "Unique values":
                st.write("Duplicate values found")
                st.dataframe(df[df.duplicated(subset=[col], keep=False)])
            
            elif rule == "Values outside allowed list":
                st.write(f"Values outside allowed list: {col_results['Fail']}")
                allowed_values = validation_rules[col][VRule.VALIDATE_LIST_OF_VALUES.value]
                st.dataframe(df[~df[col].isin(allowed_values)])
            
            elif rule == "Values not matching regex":
                st.write(f"Values not matching regex: {col_results['Fail']}")
                regex_pattern = validation_rules[col][VRule.VALIDATE_REGEX.value]
                st.dataframe(df[~df[col].str.match(regex_pattern, na=False)])
            
            elif rule == "Values out of range":
                st.write(f"Values out of range: {col_results['Fail']}")
                min_value = validation_rules[col].get(VRule.MIN_VALUE.value)
                max_value = validation_rules[col].get(VRule.MAX_VALUE.value)
                
                # Build condition based on available limits
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                condition = numeric_col.notnull()
                if min_value is not None:
                    condition &= (numeric_col < float(min_value))
                if max_value is not None:
                    condition &= (numeric_col > float(max_value))
                
                st.write(f"Range limits - Min: {min_value if min_value is not None else 'No limit'}, "
                        f"Max: {max_value if max_value is not None else 'No limit'}")
                st.dataframe(df[condition])
            
            st.write("---")

if __name__ == "__main__":
    main()

def compute_statistics(data):
    """Compute statistics for the given data.
    
    Args:
        data: Input data.
        
    Returns:
        Dictionary containing total rows, matched rows, and unmatched rows.
    """
    stats = {
        'total_rows': len(data),
        'matched_rows': str(len(data[data['matched'] == True])),  # Convert to string
        'unmatched_rows': str(len(data[data['matched'] == False]))  # Convert to string
    }
    return stats

def display_functions(df):
    """Display functions for the given dataframe.
    
    Args:
        df: Input dataframe.
        
    Returns:
        Dictionary containing column names.
    """
    return {
        Column.NAME.value: df[Column.NAME.value].tolist()
        # ...existing code...
    }

def handle_large_file(file_path, chunk_size=10000):
    """Handle large file processing in chunks.
    
    Args:
        file_path: Path to the large file.
        chunk_size: Size of each chunk.
        
    Returns:
        Concatenated dataframe.
    """
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks)

def match_data(data1, data2):
    """Match data between two sources with edge case handling.
    
    Args:
        data1: First dataframe.
        data2: Second dataframe.
        
    Returns:
        Merged dataframe.
    """
    if data1.empty or data2.empty:
        return pd.DataFrame()
    
    try:
        # Handle missing columns
        required_cols = ['A', 'name']
        for col in required_cols:
            if col not in data1.columns or col not in data2.columns:
                return pd.DataFrame()
        
        return pd.merge(data1, data2, on=['A', 'name'], how='inner')
    except Exception:
        return pd.DataFrame()

def match_records(source_df, target_df):
    """Match records between source and target dataframes.
    
    Args:
        source_df: Source dataframe.
        target_df: Target dataframe.
        
    Returns:
        Number of matched records.
    """
    if source_df.empty or target_df.empty:
        return 0
    
    try:
        matches = pd.merge(
            source_df,
            target_df,
            on=Column.NAME.value,
            how='inner'
        )
        return len(matches)
    except KeyError:
        return 0

def display_data(df: pd.DataFrame) -> None:
    """Display dataframe with proper column handling.
    
    Args:
        df: Input dataframe.
    """
    try:
        print(df[Column.NAME.value])
    except KeyError:
        print("Column not found")

def compute_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute statistics for the dataframe.
    
    Args:
        df: Input dataframe.
        
    Returns:
        Dictionary containing total, matched, unmatched, and match rate.
    """
    if df.empty:
        return {
            'total': 0,
            'matched': 0,
            'unmatched': 0,
            'match_rate': 0.0
        }
    
    total = len(df)
    matched = df['matched'].sum() if 'matched' in df.columns else 0
    return {
        'total': total,
        'matched': matched,
        'unmatched': total - matched,
        'match_rate': (matched / total) * 100 if total > 0 else 0
    }

def handle_large_file(file_path: str, chunk_size: int = 10000) -> pd.DataFrame:
    """Handle large file reading in chunks.
    
    Args:
        file_path: Path to the large file.
        chunk_size: Size of each chunk.
        
    Returns:
        Concatenated dataframe.
    """
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)

def matching_function(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Improved matching function with edge case handling.
    
    Args:
        df1: First dataframe.
        df2: Second dataframe.
        
    Returns:
        Dataframe with matched results.
    """
    if df1.empty or df2.empty:
        return pd.DataFrame()
    
    result = df1.copy()
    result['matched'] = False
    
    for col in [Column.NAME.value, 'A']:
        if col in df1.columns and col in df2.columns:
            matches = result[col].isin(df2[col])
            result.loc[matches, 'matched'] = True
    
    return result

def display_results(df: pd.DataFrame) -> None:
    """Display results with proper column handling.
    
    Args:
        df: Input dataframe.
    """
    if Column.NAME.value not in df.columns:
        return
    
    print("\nResults:")
    print(f"Total records: {len(df)}")
    stats = compute_statistics(df)
    print(f"Matched: {stats['matched']}")
    print(f"Unmatched: {stats['unmatched']}")
    print(f"Match rate: {stats['match_rate']:.2f}%")

def compute_statistics(df1, df2, matched_indices):
    """Compute statistics for the matched data.
    
    Args:
        df1: First dataframe.
        df2: Second dataframe.
        matched_indices: Indices of matched records.
        
    Returns:
        Dictionary containing total, matched, unmatched, match rate, and summary.
    """
    total = len(df1)
    matched = len(matched_indices)
    unmatched = total - matched
    match_rate = matched / total if total > 0 else 0.0
    
    return {
        'total': total,
        'matched': matched,
        'unmatched': unmatched,
        'match_rate': match_rate,
        'summary': f"Matched {matched} out of {total} records ({match_rate:.2%})"
    }

def handle_large_file(df):
    """Handle large file processing.
    
    Args:
        df: Input dataframe.
        
    Returns:
        Processed dataframe.
    """
    if len(df) > 100000:  # Define large file threshold
        return df.copy()  # Implement chunking if needed
    return df

from constants import Column

def handle_matching_edge_cases(df):
    """Handle edge cases in matching process.
    
    Args:
        df: Input dataframe.
        
    Returns:
        Processed dataframe.
    """
    if df.empty:
        return df
    
    # Handle missing values
    df = df.fillna(value='')
    
    # Handle case sensitivity
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()
    
    return df

def display_dataframe_info(df):
    """Display dataframe information with proper column handling.
    
    Args:
        df: Input dataframe.
        
    Returns:
        Dataframe information as string.
    """
    try:
        if Column.NAME in df.columns:
            # Handle name column display
            return df[Column.NAME].to_string()
    except KeyError:
        return "Column not found"
    return df.to_string()