import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import time
from typing import Union
import dask.dataframe as dd

from config import DASK_CONFIG, STREAMLIT_CONFIG, VALIDATION_CONFIG
from constants import FILE_TYPES, Column, ValidationRule as VRule, Functions
from utils import (
    FileLoader, 
    ConfigLoader, 
    execute_matching_dask,
    compute_basic_stats,
    generate_null_value_plot,
    generate_profile_report,
    display_sample
)
from state import SessionState
from connections.datahub_dask_connector import ClientDask

import asyncio
import nest_asyncio
nest_asyncio.apply()

logger = logging.getLogger(__name__)

import streamlit as st
from validators.data_validator import DataValidator
from ui.components import DataDisplay, ValidationDisplay

def main():
    st.set_page_config(page_title=STREAMLIT_CONFIG["page_title"])
    st.title("File Mapping and Validation Process")
    
    # Initialize session state variables
    SessionState.initialize()
    step = st.session_state.step

    # Create UI handler for current step
    step_handlers = {
        1: lambda: handle_file_upload("source"),
        2: lambda: handle_file_upload("target"),
        3: handle_mapping_rules,
        4: handle_matching_execution,
        5: handle_validation_rules,
        6: handle_data_validation
    }
    
    if step in step_handlers:
        step_handlers[step]()

async def async_get_tables(_client, username, token):
    from connections.datahub_dask_connector import ClientDask
    client = ClientDask(username, token)
    # Run the blocking list_tables() call in a separate thread
    return await asyncio.to_thread(client.list_tables)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_datahub_tables(username: str, token: str) -> pd.DataFrame:
    """Get tables from DataHub with caching."""
    try:
        with ClientDask(username, token) as client:
            ddf_tables = client.list_tables()
            ddf_tables['full_table_name'] = ddf_tables.map_partitions(
                lambda x: x['TABLE_SCHEMA'] + '.' + x['TABLE_NAME'],
                meta=('full_table_name', 'string')
            )
            return ddf_tables.compute()  # Convert to pandas DataFrame for caching
    except Exception as e:
        st.error(f"Error fetching tables: {str(e)}")
        return None

def handle_file_upload(file_type: str):
    st.header(f"Step {1 if file_type == 'source' else 2}: Load {file_type.capitalize()} File")
    source = st.radio("Select Source File Type", ["Local File", "DataHub"], key=f"{file_type}_source_type")
    
    if source == "Local File":
        df = load_local_file(f"{file_type}_uploader")
        if df is not None:
            SessionState.set_dataframe(f'df_{file_type}', df)
            display_metadata(df, f"{file_type.capitalize()} Data")
            if st.button("Next Step"):
                st.session_state.step += 1
                st.rerun()
    
    elif source == "DataHub":
        df = load_datahub_file(file_type)  # Pass file_type to the function
        if df is not None:
            SessionState.set_dataframe(f'df_{file_type}', df)
            display_metadata(df, f"{file_type.capitalize()} Data")
            # Note: Next step button is now handled in load_datahub_file

def load_local_file(uploader_key: str) -> Union[pd.DataFrame, None]:
    uploaded_file = st.file_uploader("Load File", type=FILE_TYPES, key=uploader_key)
    if uploaded_file:
        return FileLoader.load_file(uploaded_file)
    return None

def load_datahub_file(file_type: str) -> Union[pd.DataFrame, None]:
    username = st.text_input("Enter your DataHub username", key=f"{file_type}_username")
    token = st.text_input("Enter your DataHub token", key=f"{file_type}_token", type="password")
    query = None

    if username and token:
        try:
            # First check connectivity
            with st.expander("Connection Logs", expanded=True):
                log_placeholder = st.empty()
                client = ClientDask(username, token)
                
                # Capture and display logs
                connection_status = client.check_host_reachable(client.host, client.port)
                if not connection_status:
                    log_placeholder.error("❌ Cannot reach DataHub! Please check your VPN connection.")
                    return None
                log_placeholder.info("✓ DataHub is reachable")

            write_query = st.radio("Write your query here", ["Yes", "No"], key=f"{file_type}_write_query")
            
            if write_query == "Yes":
                query = st.text_area("Write your query here", key=f"{file_type}_query")
            else:
                with st.spinner("Gathering tables from DataHub..."):
                    try:
                        tables_df = get_datahub_tables(username, token)
                        if tables_df is not None:
                            if len(tables_df) == 0:
                                st.warning("No tables found")
                                return None
                            table = st.selectbox(
                                "Select a table to stream",
                                tables_df['full_table_name'].unique(),
                                key=f"{file_type}_table"
                            )
                            if table:
                                query = f"SELECT * FROM {table}"
                    except Exception as e:
                        st.error(f"Error connecting to DataHub: {str(e)}")
                        return None

            if query:  # Only show load button if query is defined
                col1, col2 = st.columns([2, 1])
                with col1:
                    load_button = st.button("Load Data", key=f"load_datahub_{file_type}")
                with col2:
                    next_button = st.button("Next Step", key=f"next_step_{file_type}", disabled=True)
                
                if load_button:
                    with st.spinner("Loading data from DataHub..."):
                        try:
                            with st.expander("Query Logs", expanded=True):
                                log_output = st.empty()
                            with ClientDask(username, token) as client:
                                df = client.query(query)
                                if df is not None:
                                    st.success("Data loaded successfully!")
                                    # Enable the Next Step button after successful load
                                    st.session_state[f"data_loaded_{file_type}"] = True
                                    st.rerun()  # Rerun to update the UI
                                    return df
                        except Exception as e:
                            st.error(f"Error loading data: {str(e)}")
                            logger.exception("Detailed error traceback:")
                            return None
                
                # Show enabled Next Step button if data was loaded
                if st.session_state.get(f"data_loaded_{file_type}", False):
                    if st.button("Next Step", key=f"next_step_{file_type}_enabled"):
                        st.session_state.step += 1
                        st.rerun()
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"DataHub error: {str(e)}", exc_info=True)
    return None

def stream_render_dataframe(ddf, sample_limit=100):
    """
    Streams the Dask DataFrame by computing partitions one-by-one.
    Only displays up to sample_limit rows, while accumulating the full dataset.
    """
    # Convert the Dask DataFrame into delayed partitions.
    delayed_parts = ddf.to_delayed()
    
    # Placeholder to update the UI
    placeholder = st.empty()
    accumulated_df = pd.DataFrame()
    
    st.info("Streaming results...")
    total_partitions = len(delayed_parts)
    
    for i, delayed_part in enumerate(delayed_parts):
        # Compute each partition
        partition_df = delayed_part.compute()
        accumulated_df = pd.concat([accumulated_df, partition_df], ignore_index=True)
        
        # Display only up to the sample_limit number of rows
        display_df = accumulated_df.head(sample_limit)
        placeholder.dataframe(display_df)
        st.write(f"Loaded partition {i+1}/{total_partitions}. Displaying {len(display_df)} rows (sample limit: {sample_limit}).")
        
        # Optional: small delay for visible streaming effect
        time.sleep(0.5)
    
    return accumulated_df

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
        st.rerun()

def handle_data_validation():
    st.header("Step 6: Execute Data Validation")
    df_target = st.session_state.get("df_target")
    validation_rules = st.session_state.get("validation_rules", {})
    
    if df_target is None or not validation_rules:
        st.error("Target data and validation rules are required.")
        return
    
    validator = DataValidator(validation_rules)
    try:
        results = validator.validate(df_target)
        st.session_state.validation_results = results
        
        ValidationDisplay.show_validation_results(results, df_target)
        
    except Exception as e:
        st.error(f"Validation error: {str(e)}")
    
    if st.button("Step Back"):
        st.session_state.step = 5
        st.rerun()

def display_metadata(df: Union[pd.DataFrame, dd.DataFrame], title: str):
    st.subheader(title)
    
    # Display sample data
    st.write("Data Preview:")
    st.dataframe(display_sample(df))
    
    # Compute and display basic statistics
    with st.spinner("Computing basic statistics..."):
        stats = compute_basic_stats(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", stats['row_count'])
        with col2:
            st.metric("Total Columns", stats['column_count'])
        
        st.write("Data Types:")
        st.write(pd.Series(stats['dtypes']))
        
        if 'numeric_summary' in stats:
            st.write("Statistical Summary (Numeric Columns):")
            st.dataframe(stats['numeric_summary'])
    
    # Display null values visualization
    with st.spinner("Generating null values visualization..."):
        fig = generate_null_value_plot(stats['null_counts'], title)
        st.plotly_chart(fig)
    
    # Generate and display detailed profile report
    profile_key = f"show_profile_{title.lower().replace(' ', '_')}"
    show_profile_report = st.checkbox("Show Detailed Profile Report", key=profile_key)
    
    if show_profile_report:
        try:
            with st.spinner("Generating profile report... This may take a while for large datasets."):
                profile = generate_profile_report(df, title)
                st.components.v1.html(profile.to_html(), height=600, scrolling=True)
        except Exception as e:
            st.error(f"Error generating profile report: {str(e)}")
            logger.exception("Profile report generation failed:")

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
                max_val = validation_rules.col.get(VRule.MAX_VALUE.value)
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

import logging
import streamlit as st
from state import SessionState
from ui.pages import (
    FileUploadPage,
    MappingPage,
    MatchingPage,
    ValidationPage
)
from utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

class App:
    def __init__(self):
        self.pages = {
            1: FileUploadPage("source"),
            2: FileUploadPage("target"),
            3: MappingPage(),
            4: MatchingPage(),
            5: ValidationPage()
        }
        setup_logging()

    def run(self):
        st.set_page_config(page_title="File Mapping and Validation Process")
        st.title("File Mapping and Validation Process")
        
        SessionState.initialize()
        current_step = st.session_state.step
        
        try:
            if current_step in self.pages:
                self.pages[current_step].render()
        except Exception as e:
            logger.error(f"Error in step {current_step}: {str(e)}", exc_info=True)
            st.error(f"An error occurred: {str(e)}")

def main():
    app = App()
    app.run()

if __name__ == "__main__":
    main()