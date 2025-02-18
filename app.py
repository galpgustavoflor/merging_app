import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import json

from config import DASK_CONFIG, STREAMLIT_CONFIG, VALIDATION_CONFIG
from constants import FILE_TYPES, Column, ValidationRule as VRule, Functions
from utils import FileLoader, ConfigLoader, DataValidator, execute_matching_dask
from state import SessionState

logger = logging.getLogger(__name__)

def main():
    st.set_page_config(page_title=STREAMLIT_CONFIG["page_title"])
    st.title("File Mapping and Validation Process")
    
    # Initialize session state variables
    SessionState.initialize()
    step = st.session_state.step

    if step == 1:
        handle_source_file_upload()
    elif step == 2:
        handle_target_file_upload()
    elif step == 3:
        handle_mapping_rules()
    elif step == 4:
        handle_matching_execution()
    elif step == 5:
        handle_validation_rules()
    elif step == 6:
        handle_data_validation()

def handle_source_file_upload():
    st.header("Step 1: Load Source File")
    uploaded_source = st.file_uploader("Load Source File", type=FILE_TYPES, key="source_uploader")
    if uploaded_source:
        df = FileLoader.load_file(uploaded_source)
        if df is not None:
            SessionState.set_dataframe('df_source', df)
            display_metadata(df, "Source Data")
            if st.button("Next Step"):
                st.session_state.step = 2
                st.rerun()

def handle_target_file_upload():
    st.header("Step 2: Load Target File")
    uploaded_target = st.file_uploader("Load Target File", type=FILE_TYPES, key="target_uploader")
    if uploaded_target:
        df = FileLoader.load_file(uploaded_target)
        if df is not None:
            SessionState.set_dataframe('df_target', df)
            display_metadata(df, "Target Data")
            if st.button("Next Step"):
                st.session_state.step = 3
                st.rerun()

def handle_mapping_rules():
    st.header("Step 3: Define Mapping Rules and Keys")
    
    with st.expander("Upload JSON File for Mapping"):
        uploaded_json = st.file_uploader("Load Mapping Rules from JSON", type=["json"], key="mapping_json")
        if uploaded_json:
            ConfigLoader.load_mapping_from_json(uploaded_json)
    
    df_source = st.session_state.get("df_source")
    df_target = st.session_state.get("df_target")
    if df_source is None or df_target is None:
        st.error("Please upload both source and target files.")
        return
    
    key_source = st.multiselect("Select the key(s) in the source", df_source.columns.tolist(), default=st.session_state.mapping.get("key_source", []))
    key_target = st.multiselect("Select the key(s) in the target", df_target.columns.tolist(), default=st.session_state.mapping.get("key_target", []))
    
    st.session_state.key_source = key_source
    st.session_state.key_target = key_target
    
    mapping_config = st.session_state.get("mapping", {})
    for col in df_source.columns:
        if col not in key_source:
            with st.expander(f"Configure mapping for '{col}'"):
                option = st.radio(f"Action for column '{col}'", ["Ignore", "Map"], key=f"option_{col}", index=1 if col in mapping_config.get("mappings", {}) else 0)
                if option == "Map":
                    mapped_cols = st.multiselect(f"Map '{col}' to:", df_target.columns.tolist(), key=f"map_{col}", default=mapping_config.get("mappings", {}).get(col, {}).get("destinations", []))
                    function = st.selectbox(f"Mapping Type for '{col}'", [f.value for f in Functions], key=f"func_{col}", index=0)
                    transformation = None
                    if function == Functions.AGGREGATION.value:
                        transformation = st.selectbox("Aggregation Type", ["Sum", "Mean", "Median", "Max", "Min"], key=f"agg_{col}")
                    elif function == Functions.CONVERSION.value:
                        transformation = st.text_area("Define Conversion Dictionary (JSON)", "{}", key=f"conv_{col}")
                        try:
                            json.loads(transformation)
                        except json.JSONDecodeError:
                            st.error("Invalid JSON format for conversion dictionary.")
                    mapping_config.setdefault("mappings", {})[col] = {
                        "destinations": mapped_cols,
                        "function": function,
                        "transformation": transformation
                    }
    mapping_config["key_source"] = key_source
    mapping_config["key_target"] = key_target
    st.session_state.mapping = mapping_config
    
    if st.checkbox("Show Mapping Configuration (JSON)"):
        st.json(st.session_state.mapping)
    
    if st.button("Next Step"):
        st.session_state.step = 4
        st.rerun()

def handle_matching_execution():
    st.header("Step 4: Execute Matching")
    key_source = st.session_state.get("key_source")
    key_target = st.session_state.get("key_target")
    df_source = st.session_state.get("df_source")
    df_target = st.session_state.get("df_target")
    
    if not key_source or not key_target:
        st.error("Please select valid keys for matching.")
        return
    
    ddf_merged, stats = execute_matching_dask(df_source, df_target, key_source, key_target)
    
    st.subheader("Matching Summary")
    st.write(f"Total matched records: {stats['total_match']}")
    st.write(f"Missing in source: {stats['missing_source']}")
    st.write(f"Missing in target: {stats['missing_target']}")
    
    matched_df = ddf_merged[ddf_merged['_merge'] == 'both'].compute()
    st.session_state.matching_results = matched_df
    
    sample_size = min(DASK_CONFIG["sample_size"], len(ddf_merged))
    sample_df = ddf_merged.compute().sample(n=sample_size, random_state=42)
    
    st.subheader("Matching Records (Sample)")
    st.dataframe(sample_df[sample_df['_merge'] == 'both'])
    
    st.subheader("Non-Matching Records (Sample)")
    non_matching = sample_df[sample_df['_merge'] != 'both']
    st.dataframe(non_matching)
    
    if st.button("Download Full Non-Matching Records"):
        full_non_matching = ddf_merged[ddf_merged['_merge'] != 'both'].compute()
        csv = full_non_matching.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="non_matching_records.csv", mime="text/csv")
    
    if st.button("Step Back"):
        st.session_state.step = 3
        st.rerun()
    
    if st.button("Configure Data Validations"):
        st.session_state.step = 5
        st.rerun()

def handle_validation_rules():
    st.header("Step 5: Define Validation Rules")
    df_target = st.session_state.get("df_target")
    if df_target is None:
        st.error("Please load the target file first.")
        return
    
    with st.expander("Upload JSON File for Validation Rules"):
        uploaded_json = st.file_uploader("Load Validation Rules from JSON", type=["json"], key="validation_json")
        if uploaded_json:
            ConfigLoader.load_validations_from_json(uploaded_json)
    
    validation_rules = st.session_state.get("validation_rules", {})
    for col in df_target.columns:
        with st.expander(f"Configure validation for '{col}'"):
            if st.checkbox("Check Nulls", key=f"nulls_{col}", value=validation_rules.get(col, {}).get(VRule.VALIDATE_NULLS.value, False)):
                validation_rules.setdefault(col, {})[VRule.VALIDATE_NULLS.value] = True
            if st.checkbox("Check Uniqueness", key=f"unique_{col}", value=validation_rules.get(col, {}).get(VRule.VALIDATE_UNIQUENESS.value, False)):
                validation_rules.setdefault(col, {})[VRule.VALIDATE_UNIQUENESS.value] = True
            if st.checkbox("Check Allowed Values", key=f"domain_{col}", value=bool(validation_rules.get(col, {}).get(VRule.VALIDATE_LIST_OF_VALUES.value))):
                allowed_values = st.text_input("Allowed Values (comma separated)", key=f"domain_values_{col}", value=",".join(validation_rules.get(col, {}).get(VRule.VALIDATE_LIST_OF_VALUES.value, [])))
                validation_rules.setdefault(col, {})[VRule.VALIDATE_LIST_OF_VALUES.value] = [val.strip() for val in allowed_values.split(',') if val.strip()]
            if st.checkbox("Check Format (Regex)", key=f"regex_{col}", value=bool(validation_rules.get(col, {}).get(VRule.VALIDATE_REGEX.value))):
                regex_pattern = st.text_input("Regex Pattern", key=f"regex_pattern_{col}", value=validation_rules.get(col, {}).get(VRule.VALIDATE_REGEX.value, ""))
                validation_rules.setdefault(col, {})[VRule.VALIDATE_REGEX.value] = regex_pattern
            if pd.api.types.is_numeric_dtype(df_target[col]):
                if st.checkbox("Check Range", key=f"range_{col}", value=bool(validation_rules.get(col, {}).get(VRule.VALIDATE_RANGE.value))):
                    validation_rules.setdefault(col, {})[VRule.VALIDATE_RANGE.value] = True
                    has_min = st.checkbox("Set Minimum", key=f"has_min_{col}", value=validation_rules.get(col, {}).get(VRule.MIN_VALUE.value) is not None)
                    if has_min:
                        min_value = st.number_input("Minimum Value", key=f"min_{col}", value=validation_rules.get(col, {}).get(VRule.MIN_VALUE.value, 0.0))
                        validation_rules[col][VRule.MIN_VALUE.value] = min_value
                    else:
                        validation_rules[col][VRule.MIN_VALUE.value] = None
                    has_max = st.checkbox("Set Maximum", key=f"has_max_{col}", value=validation_rules.get(col, {}).get(VRule.MAX_VALUE.value) is not None)
                    if has_max:
                        max_value = st.number_input("Maximum Value", key=f"max_{col}", value=validation_rules.get(col, {}).get(VRule.MAX_VALUE.value, 0.0))
                        validation_rules[col][VRule.MAX_VALUE.value] = max_value
                    else:
                        validation_rules[col][VRule.MAX_VALUE.value] = None
    st.session_state.validation_rules = validation_rules
    
    if st.checkbox("Show Validation Configuration (JSON)"):
        st.json(validation_rules)
    
    if st.button("Next Step"):
        st.session_state.step = 6
        #st.experimental_rerun()

def handle_data_validation():
    st.header("Step 6: Execute Data Validation")
    df_target = st.session_state.get("df_target")
    validation_rules = st.session_state.get("validation_rules", {})
    if df_target is None or not validation_rules:
        st.error("Target data and validation rules are required.")
        return
    
    results = DataValidator.execute_validation(df_target, validation_rules)
    st.session_state.validation_results = results
    
    display_validation_summary(results)
    display_detailed_validation_results(df_target, results, validation_rules)
    
    if st.button("Step Back"):
        st.session_state.step = 5
        st.experimental_rerun()

def display_metadata(df: pd.DataFrame, title: str):
    st.subheader(title)
    st.write("Data Preview:")
    st.dataframe(df.head())
    st.write("Statistical Summary:")
    st.write(df.describe(include='all'))
    st.write("Data Types:")
    st.write(df.dtypes)
    st.write("Null Values by Column:")
    null_counts = df.isnull().sum().reset_index()
    null_counts.columns = [df.columns[0] if len(df.columns)>0 else "Column", "Null Count"]
    fig = px.bar(null_counts, x=null_counts.columns[0], y="Null Count", title="Null Values by Column")
    st.plotly_chart(fig)

def display_validation_summary(validation_results: list):
    st.subheader("Validation Results Summary")
    summary_df = pd.DataFrame(validation_results)
    def color_pass(val):
        if isinstance(val, str) and val.endswith('%'):
            percentage = float(val.rstrip('%'))
            return 'background-color: green' if percentage >= VALIDATION_CONFIG["pass_threshold"] else 'background-color: red'
        return ''
    st.dataframe(summary_df.style.applymap(color_pass))

def display_detailed_validation_results(df: pd.DataFrame, validation_results: list, validation_rules: dict):
    st.subheader("Detailed Validation Results")
    for result in validation_results:
        if result["Fail"] > 0:
            col = result[Column.NAME.value]
            st.write(f"### Column: {col}")
            rule = result["Rule"]
            if rule == "Null values":
                st.write(f"Null count: {result['Fail']}")
                st.dataframe(df[df[col].isnull()])
            elif rule == "Unique values":
                st.write("Duplicate values found:")
                st.dataframe(df[df.duplicated(subset=[col], keep=False)])
            elif rule == "Values outside allowed list":
                st.write(f"Values not in allowed list: {result['Fail']}")
                allowed = validation_rules[col][VRule.VALIDATE_LIST_OF_VALUES.value]
                st.dataframe(df[~df[col].isin(allowed)])
            elif rule == "Values not matching regex":
                st.write(f"Regex mismatch count: {result['Fail']}")
                regex_pattern = validation_rules[col][VRule.VALIDATE_REGEX.value]
                st.dataframe(df[~df[col].astype(str).str.match(regex_pattern, na=False)])
            elif rule == "Values out of range":
                st.write(f"Out of range count: {result['Fail']}")
                min_val = validation_rules[col].get(VRule.MIN_VALUE.value)
                max_val = validation_rules[col].get(VRule.MAX_VALUE.value)
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                condition = numeric_col.notnull()
                if min_val is not None:
                    condition &= (numeric_col < float(min_val))
                if max_val is not None:
                    condition &= (numeric_col > float(max_val))
                st.dataframe(df[condition])
            st.write("---")

if __name__ == "__main__":
    main()