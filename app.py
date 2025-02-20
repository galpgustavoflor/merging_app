import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import hashlib  # add import at top if not already imported
import datetime  # <-- new import for timestamp
from fpdf import FPDF  # new import for PDF generation

from config import DASK_CONFIG, STREAMLIT_CONFIG, VALIDATION_CONFIG
from constants import FILE_TYPES, Column, ValidationRule as VRule, Functions, Step, STEP_LABELS
from utils import FileLoader, ConfigLoader, DataValidator, execute_matching_dask, clean_dataframe_for_display
from state import SessionState

logger = logging.getLogger(__name__)

def inject_custom_css():
    """Inject custom CSS for modern styling."""
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def render_breadcrumbs():
    with st.container():
        # Removed aggressive white background and padding for a minimalistic look
        st.markdown('<div class="breadcrumb-container" style="display: flex; justify-content: center; gap: 1em; margin-bottom: 1em;">', unsafe_allow_html=True)
        cols = st.columns(len(STEP_LABELS))
        for i, (step, label) in enumerate(STEP_LABELS.items()):
            with cols[i]:
                if step.value < st.session_state.step:
                    if st.button(f"✔ {label}", key=f"bread_{step}", help="Go to this step"):
                        st.session_state.step = step.value
                        st.rerun()  # updated rerun call
                elif step.value == st.session_state.step:
                    st.markdown(f'<div class="breadcrumb-active" style="color: #FF6600; font-weight: bold;">● {label}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="breadcrumb-pending" style="color: #B0BEC5;">○ {label}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def step_navigation(back=True, next=True, next_label="Next Step"):
    """Render navigation buttons."""
    cols = st.columns(2)
    if back and st.session_state.step > Step.SOURCE_UPLOAD.value:
        if cols[0].button("← Back", key="back_button", help="Go to previous step", use_container_width=True):
            st.session_state.step -= 1
            st.rerun()  # updated rerun call
    if next:
        if cols[1].button(f"{next_label} →", key="next_button", help="Proceed to next step", use_container_width=True):
            st.session_state.step += 1
            st.rerun()  # updated rerun call

def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title=STREAMLIT_CONFIG["page_title"],
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="🍊"  # Updated to reflect the orange branding
    )
    
    # Initialize session state variables
    if 'step' not in st.session_state:
        st.session_state.step = Step.SOURCE_UPLOAD.value
    if 'df_source' not in st.session_state:
        st.session_state.df_source = None
    if 'df_target' not in st.session_state:
        st.session_state.df_target = None
    if 'mapping' not in st.session_state:
        st.session_state.mapping = {}
    if 'validation_rules' not in st.session_state:
        st.session_state.validation_rules = {}
    if 'matching_results' not in st.session_state:
        st.session_state.matching_results = None
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = None

    inject_custom_css()
    
    with st.container():
        # Updated modern header with orange accent and futuristic typography
        st.markdown(
            '<h1 class="main-title" style="text-align: center; font-size: 3em; margin-bottom: 0.5em; color: #FF6600; font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif;">'
            'File Mapping &amp; Validation'
            '</h1>',
            unsafe_allow_html=True
        )
    
    render_breadcrumbs()
    
    with st.container():
        st.markdown('<div class="content-card">', unsafe_allow_html=True)
        # Handle steps
        step_handlers = {
            Step.SOURCE_UPLOAD: handle_source_file_upload,
            Step.TARGET_UPLOAD: handle_target_file_upload,
            Step.MAPPING_RULES: handle_mapping_rules,
            Step.MATCHING: handle_matching_execution,
            Step.VALIDATION_RULES: handle_validation_rules,
            Step.DATA_VALIDATION: handle_data_validation,
            Step.REPORT_SUMMARY: handle_report_summary  # Added new final step
        }
        
        current_handler = step_handlers.get(Step(st.session_state.step))
        if current_handler:
            current_handler()
        st.markdown('</div>', unsafe_allow_html=True)

def handle_source_file_upload():
    st.markdown('<h2 class="step-header">Step 1: Load Source File</h2>', unsafe_allow_html=True)
    uploaded_source = st.file_uploader("Load Source File", type=FILE_TYPES, key="source_uploader")
    
    if uploaded_source:
        df = FileLoader.load_file(uploaded_source)
        if df is not None:
            SessionState.set_dataframe('df_source', df)
            display_metadata(df, "Source Data")
            step_navigation(back=False)

def handle_target_file_upload():
    st.markdown('<h2 class="step-header">Step 2: Load Target File</h2>', unsafe_allow_html=True)
    uploaded_target = st.file_uploader("Load Target File", type=FILE_TYPES, key="target_uploader")
    
    if uploaded_target:
        df = FileLoader.load_file(uploaded_target)
        if df is not None:
            SessionState.set_dataframe('df_target', df)
            display_metadata(df, "Target Data")
            step_navigation()

def handle_mapping_rules():
    st.header("Step 3: Define Mapping Rules and Keys")
    
    # Initialize mapping in session state if not exists
    if "mapping" not in st.session_state:
        st.session_state.mapping = {"mappings": {}, "key_source": [], "key_target": []}
    
    # Handle JSON upload with safe state management
    with st.expander("Upload JSON File for Mapping"):
        uploaded_json = st.file_uploader(
            "Load Mapping Rules from JSON", 
            type=["json"], 
            key="mapping_json",
            help="Upload a JSON file containing mapping rules"
        )
        
        # Store the file upload state
        if "previous_upload" not in st.session_state:
            st.session_state.previous_upload = None
            
        # Only process if there's a new file and it's different from the previous one
        if uploaded_json and uploaded_json != st.session_state.previous_upload:
            try:
                mapping_data = json.load(uploaded_json)
                # Validate the mapping structure before updating
                if isinstance(mapping_data, dict) and "mappings" in mapping_data:
                    st.session_state.mapping = mapping_data
                    st.session_state.previous_upload = uploaded_json
                    st.success("Mapping rules loaded successfully!")
                else:
                    st.error("Invalid mapping file structure")
            except Exception as e:
                st.error(f"Error loading mapping file: {str(e)}")
    
    # Validate required data
    df_source = st.session_state.get("df_source")
    df_target = st.session_state.get("df_target")
    if df_source is None or df_target is None:
        st.error("Please upload both source and target files.")
        return
    
    # Create a working copy of mapping config
    mapping_config = st.session_state.mapping.copy()
    
    # Key selection with defaults from current mapping
    col1, col2 = st.columns(2)
    with col1:
        key_source = st.multiselect(
            "Select source key(s)", 
            df_source.columns.tolist(),
            default=mapping_config.get("key_source", []),
            key="key_source_select"
        )
    with col2:
        key_target = st.multiselect(
            "Select target key(s)",
            df_target.columns.tolist(),
            default=mapping_config.get("key_target", []),
            key="key_target_select"
        )
    
    # Update keys in mapping config
    mapping_config["key_source"] = key_source
    mapping_config["key_target"] = key_target
    
    # Column mapping configuration
    st.subheader("Column Mappings")
    mapped_columns = set()  # Track mapped columns
    
    for col in df_source.columns:
        if col not in key_source:  # Skip key columns
            with st.expander(f"Configure mapping for '{col}'", expanded=col in mapping_config.get("mappings", {})):
                col_config = mapping_config.get("mappings", {}).get(col, {})
                
                # Map column or ignore
                map_col = st.checkbox(
                    f"Map column '{col}'",
                    value=col in mapping_config.get("mappings", {}),
                    key=f"map_checkbox_{col}"
                )
                
                if map_col:
                    # Destination column selection
                    mapped_cols = st.multiselect(
                        f"Map '{col}' to:",
                        df_target.columns.tolist(),
                        default=col_config.get("destinations", []),
                        key=f"dest_select_{col}"
                    )
                    
                    # Mapping function selection
                    function = st.selectbox(
                        "Mapping Type",
                        [f.value for f in Functions],
                        index=[f.value for f in Functions].index(col_config.get("function", Functions.DIRECT.value)),
                        key=f"function_select_{col}"
                    )
                    
                    # Handle transformation based on function
                    transformation = None
                    if function == Functions.CONVERSION.value:
                        # Create two columns for the conversion mapping interface
                        conv_col1, conv_col2 = st.columns([2, 3])
                        
                        with conv_col1:
                            # JSON input
                            transformation = st.text_area(
                                "Conversion Dictionary (JSON)",
                                value=col_config.get("transformation", "{}"),
                                key=f"transform_input_{col}",
                                height=200
                            )
                            
                            # Add help text
                            st.info("Example: {'source_value': 'target_value'}")
                            
                            try:
                                # Validate and parse JSON
                                transformation_dict = json.loads(transformation or "{}")
                                if not isinstance(transformation_dict, dict):
                                    st.error("Transformation must be a valid dictionary")
                                    transformation = "{}"
                            except json.JSONDecodeError:
                                st.error("Invalid JSON format")
                                transformation = "{}"
                        
                        with conv_col2:
                            # Visual mapping table
                            st.write("Visual Mapping Table")
                            
                            # Get unique values from source column
                            unique_values = df_source[col].unique()
                            
                            # Create mapping table
                            mapping_data = []
                            try:
                                current_mappings = json.loads(transformation or "{}")
                                for val in unique_values:
                                    mapping_data.append({
                                        "Source Value": str(val),
                                        "Target Value": current_mappings.get(str(val), ""),
                                        "Sample Count": int(df_source[col].eq(val).sum())
                                    })
                                
                                # Display as DataFrame with edit functionality
                                mapping_df = pd.DataFrame(mapping_data)
                                edited_df = st.data_editor(
                                    mapping_df,
                                    key=f"mapping_table_{col}",
                                    hide_index=True,
                                    use_container_width=True,
                                    num_rows="fixed"
                                )
                                
                                # Update JSON when table is edited
                                if not edited_df.equals(mapping_df):
                                    new_mapping = {
                                        str(row["Source Value"]): str(row["Target Value"])
                                        for _, row in edited_df.iterrows()
                                        if row["Target Value"]  # Only include non-empty mappings
                                    }
                                    transformation = json.dumps(new_mapping, indent=2)
                            except Exception as e:
                                st.error(f"Error creating mapping table: {str(e)}")
                            
                            # Add statistics
                            st.write("Value Distribution")
                            value_counts = df_source[col].value_counts().head(10)
                            st.bar_chart(value_counts)
                    
                    elif function == Functions.AGGREGATION.value:
                        transformation = st.selectbox(
                            "Aggregation Function",
                            ["sum", "mean", "median", "max", "min"],
                            index=0,
                            key=f"agg_select_{col}"
                        )
                    
                    # Update mapping configuration
                    if mapped_cols:
                        mapping_config.setdefault("mappings", {})[col] = {
                            "destinations": mapped_cols,
                            "function": function,
                            "transformation": transformation
                        }
                        mapped_columns.add(col)
                
                # Remove unmapped column
                elif col in mapping_config.get("mappings", {}):
                    del mapping_config["mappings"][col]
    
    # Clean up any orphaned mappings
    if "mappings" in mapping_config:
        orphaned = set(mapping_config["mappings"].keys()) - mapped_columns
        for col in orphaned:
            del mapping_config["mappings"][col]
    
    # Update session state
    st.session_state.mapping = mapping_config
    
    # Display current mapping configuration
    if st.checkbox("Show Mapping Configuration", key="show_mapping"):
        st.json(mapping_config)
    
    # Navigation
    if key_source and key_target and mapping_config.get("mappings"):
        step_navigation()
    else:
        st.warning("Please configure at least one mapping and select keys before proceeding.")
        step_navigation(next=False)

def handle_matching_execution():
    st.header("Step 4: Execute Matching")
    
    # Get mapping configuration
    mapping_config = st.session_state.get("mapping")
    if not mapping_config:
        st.error("Please define mapping rules first.")
        step_navigation(next=False)
        return
    
    # Get data from session state
    df_source = st.session_state.get("df_source")
    df_target = st.session_state.get("df_target")
    
    if df_source is None or df_target is None:
        st.error("Please upload both source and target files.")
        return
    
    try:
        # Execute matching with mapping config
        ddf_merged, stats = execute_matching_dask(df_source, df_target, mapping_config)
        
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
    
    except Exception as e:
        st.error(f"Error during matching execution: {str(e)}")
        logger.error(f"Matching execution error: {e}", exc_info=True)
        step_navigation(next=False)
        return
        
    step_navigation(next_label="Configure Validations")

def handle_validation_rules():
    st.header("Step 5: Define Validation Rules")
    
    # Fixed check for matching results before proceeding
    if st.session_state.get("matching_results") is None:
        st.error("Please complete the matching step first.")
        step_navigation(next=False)
        return
    
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
    
    step_navigation()

def handle_data_validation():
    st.header("Step 6: Execute Data Validation")
    
    if not st.session_state.get("validation_rules"):
        st.error("Please define validation rules first.")
        step_navigation(next=False)
        return
    
    df_target = st.session_state.get("df_target")
    validation_rules = st.session_state.get("validation_rules", {})
    if df_target is None or not validation_rules:
        st.error("Target data and validation rules are required.")
        return
    
    results = DataValidator.execute_validation(df_target, validation_rules)
    st.session_state.validation_results = results
    
    display_validation_summary(results)
    display_detailed_validation_results(df_target, results, validation_rules)
    
    step_navigation()

def handle_report_summary():
    st.header("Final Report Summary for Audit")
    
    # Data Audit Summary with SHA256 for file integrity (market practice)
    st.subheader("Data Audit Summary")
    checksum_note = "SHA256 hash computed from the CSV representation of the dataset."
    source_df = st.session_state.get("df_source")
    target_df = st.session_state.get("df_target")
    
    source_summary = {}
    target_summary = {}
    
    if source_df is not None:
        csv_source = source_df.to_csv(index=False).encode('utf-8')
        source_hash = hashlib.sha256(csv_source).hexdigest()
        st.write(f"Source Dataset: {source_df.shape[0]} rows, {source_df.shape[1]} columns.")
        st.write(f"Checksum (SHA256): {source_hash}")
        source_summary = {
            "rows": source_df.shape[0],
            "columns": source_df.shape[1],
            "checksum": source_hash
        }
    else:
        st.info("No source dataset available.")
    
    if target_df is not None:
        csv_target = target_df.to_csv(index=False).encode('utf-8')
        target_hash = hashlib.sha256(csv_target).hexdigest()
        st.write(f"Target Dataset: {target_df.shape[0]} rows, {target_df.shape[1]} columns.")
        st.write(f"Checksum (SHA256): {target_hash}")
        target_summary = {
            "rows": target_df.shape[0],
            "columns": target_df.shape[1],
            "checksum": target_hash
        }
    else:
        st.info("No target dataset available.")
    
    st.info(f"***Checksum Note:*** {checksum_note}", icon=":material/info:")
    
    # Display mapping configuration if available
    st.subheader("Mapping Configuration")
    mapping = st.session_state.get("mapping", {})  
    if mapping:
        st.write(f"Source Key(s): {', '.join(mapping.get('key_source', []))}")
        st.write(f"Target Key(s): {', '.join(mapping.get('key_target', []))}")
        mapping_list = []
        for col, config in mapping.get("mappings", {}).items():
            mapping_list.append({
                "Column": col,
                "Destinations": ", ".join(config.get("destinations", [])),
                "Function": config.get("function", ""),
                "Transformation": config.get("transformation", "")
            })
        df_mapping = pd.DataFrame(mapping_list)
        st.dataframe(df_mapping)
        with st.expander("Mapping Configuration JSON", expanded=False):
            st.json(mapping)
    else:
        st.info("No mapping defined.")
    
    
    
    # Display matching results summary
    st.subheader("Matching Results")
    matching_results = st.session_state.get("matching_results")
    if matching_results is not None:
        st.write(f"Matched Records: {len(matching_results)}")
        st.dataframe(matching_results)
    else:
        st.info("No matching results available.")
    
    # Display validation configuration
    st.subheader("Validation Configuration")
    validation_config = st.session_state.get("validation_rules", {})
    if validation_config:
        try:
            df_validation = pd.DataFrame.from_dict(validation_config, orient='index')
            st.dataframe(df_validation)
        except Exception:
            st.write("Validation Configuration:")
            st.json(validation_config)
        with st.expander("Validation Configuration JSON", expanded=False):
            st.json(validation_config)
    else:
        st.info("No validation configuration defined.")
    
    # Display validation results summary, if any
    st.subheader("Validation Results Summary")
    validation_results = st.session_state.get("validation_results")
    if validation_results:
        st.write(f"Total Validations: {len(validation_results)}")
        display_validation_summary(validation_results)
        display_detailed_validation_results(
            st.session_state.get("df_target"),
            validation_results,
            st.session_state.get("validation_rules", {})
        )
    else:
        st.info("No validation results available.")
    
    # Prepare a downloadable report (JSON content used for PDF)
    report = {
        "data_audit": {
            "source": source_summary,
            "target": target_summary
        },
        "mapping": mapping,
        "validation_configuration": validation_config,
        "matching_results_count": int(len(matching_results) if matching_results is not None else 0),
        "validation_results": validation_results
    }
    report_json = json.dumps(report, indent=4, default=lambda o: int(o) if isinstance(o, (np.integer)) else o)
    # Compute report checksum and timestamp
    report_checksum = hashlib.sha256(report_json.encode('utf-8')).hexdigest()
    timestamp = datetime.datetime.now().isoformat()
    
    st.markdown(f"<div style='margin-top: 1em;'><strong>Report Generated At:</strong> {timestamp}</div>", unsafe_allow_html=True)
    st.markdown(f"<div><strong>Report Checksum (SHA256):</strong> {report_checksum}</div>", unsafe_allow_html=True)
    
    # Generate PDF report using report_json as content
    pdf_data = generate_pdf_report(report_json, source_summary, target_summary, checksum_note, mapping, matching_results, validation_config, validation_results)
    st.download_button("Download Audit Report (PDF)", data=pdf_data, file_name="audit_report.pdf", mime="application/pdf")
    
    step_navigation(next=False)

def generate_pdf_report(content: str, source_summary: dict, target_summary: dict, checksum_note: str, mapping: dict, matching_results: pd.DataFrame, validation_config: dict, validation_results: list) -> bytes:
    # Create a PDF report with a corporate design
    pdf = FPDF()
    pdf.add_page()
    # Corporate header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Corporate Audit Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Confidential", ln=True, align="C")
    pdf.ln(10)
    # Report content using same information from the last step
    pdf.set_font("Arial", "", 10)
    for line in content.splitlines():
        pdf.multi_cell(0, 8, line)
    # Add footer with timestamp and checksum
    pdf.ln(10)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, f"Report Generated At: {datetime.datetime.now().isoformat()}", ln=True, align="L")
    pdf.cell(0, 10, f"Report Checksum (SHA256): {hashlib.sha256(content.encode('utf-8')).hexdigest()}", ln=True, align="L")
    
    # Add graphics and styling to match the last page
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Data Audit Summary", ln=True, align="L")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Source Dataset:", ln=True, align="L")
    pdf.cell(0, 10, f"Rows: {source_summary['rows']}, Columns: {source_summary['columns']}", ln=True, align="L")
    pdf.cell(0, 10, f"Checksum (SHA256): {source_summary['checksum']}", ln=True, align="L")
    pdf.ln(5)
    pdf.cell(0, 10, "Target Dataset:", ln=True, align="L")
    pdf.cell(0, 10, f"Rows: {target_summary['rows']}, Columns: {target_summary['columns']}", ln=True, align="L")
    pdf.cell(0, 10, f"Checksum (SHA256): {target_summary['checksum']}", ln=True, align="L")
    pdf.ln(10)
    pdf.cell(0, 10, f"Checksum Note: {checksum_note}", ln=True, align="L")
    
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Mapping Configuration", ln=True, align="L")
    pdf.set_font("Arial", "", 12)
    if mapping:
        pdf.cell(0, 10, f"Source Key(s): {', '.join(mapping.get('key_source', []))}", ln=True, align="L")
        pdf.cell(0, 10, f"Target Key(s): {', '.join(mapping.get('key_target', []))}", ln=True, align="L")
        pdf.ln(5)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 10, "Column", 1)
        pdf.cell(60, 10, "Destinations", 1)
        pdf.cell(40, 10, "Function", 1)
        pdf.cell(50, 10, "Transformation", 1)
        pdf.ln()
        pdf.set_font("Arial", "", 12)
        for col, config in mapping.get("mappings", {}).items():
            pdf.cell(40, 10, col, 1)
            pdf.cell(60, 10, ", ".join(config.get("destinations", [])), 1)
            pdf.cell(40, 10, config.get("function", ""), 1)
            pdf.cell(50, 10, str(config.get("transformation", "")), 1)
            pdf.ln()
    else:
        pdf.cell(0, 10, "No mapping defined.", ln=True, align="L")
    
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Matching Results", ln=True, align="L")
    pdf.set_font("Arial", "", 12)
    if matching_results is not None:
        pdf.cell(0, 10, f"Matched Records: {len(matching_results)}", ln=True, align="L")
        # Add a sample of matching results
        for index, row in matching_results.head(10).iterrows():
            pdf.cell(0, 10, str(row.to_dict()), ln=True, align="L")
            pdf.ln(5)
    else:
        pdf.cell(0, 10, "No matching results available.", ln=True, align="L")
    
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Validation Configuration", ln=True, align="L")
    pdf.set_font("Arial", "", 12)
    if validation_config:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 10, "Column", 1)
        pdf.cell(60, 10, "Rule", 1)
        pdf.cell(90, 10, "Value", 1)
        pdf.ln()
        pdf.set_font("Arial", "", 12)
        for col, rules in validation_config.items():
            for rule, value in rules.items():
                pdf.cell(40, 10, col, 1)
                pdf.cell(60, 10, rule, 1)
                pdf.cell(90, 10, str(value), 1)
                pdf.ln()
    else:
        pdf.cell(0, 10, "No validation configuration defined.", ln=True, align="L")
    
    pdf.add_page()
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Validation Results Summary", ln=True, align="L")
    pdf.set_font("Arial", "", 12)
    if validation_results:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(40, 10, "Column", 1)
        pdf.cell(60, 10, "Rule", 1)
        pdf.cell(40, 10, "Pass", 1)
        pdf.cell(40, 10, "Fail", 1)
        pdf.ln()
        pdf.set_font("Arial", "", 12)
        for result in validation_results:
            pdf.cell(40, 10, result[Column.NAME.value], 1)
            pdf.cell(60, 10, result["Rule"], 1)
            pdf.cell(40, 10, str(result["Pass"]), 1)
            pdf.cell(40, 10, str(result["Fail"]), 1)
            pdf.ln()
    else:
        pdf.cell(0, 10, "No validation results available.", ln=True, align="L")
    
    return pdf.output(dest="S").encode("latin1")

def display_metadata(df: pd.DataFrame, title: str):
    st.markdown(f'<h3 class="step-header">{title}</h3>', unsafe_allow_html=True)
    
    try:
        display_df = clean_dataframe_for_display(df)
        
        tabs = st.tabs(["Preview", "Summary", "Data Types", "Null Analysis"])
        
        with tabs[0]:
            st.markdown('<div class="data-preview">', unsafe_allow_html=True)
            st.dataframe(display_df.head(), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[1]:
            st.markdown('<div class="data-preview">', unsafe_allow_html=True)
            summary_df = clean_dataframe_for_display(df.describe(include='all'))
            st.dataframe(summary_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[2]:
            st.markdown('<div class="data-preview">', unsafe_allow_html=True)
            st.write(display_df.dtypes)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[3]:
            null_counts = display_df.isnull().sum().reset_index()
            null_counts.columns = ["Column", "Null Count"]
            fig = px.bar(
                null_counts,
                x="Column",
                y="Null Count",
                title="Null Values Distribution",
                template="plotly_white"
            )
            fig.update_layout(
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error displaying data: {str(e)}")
        logger.error(f"Error in display_metadata: {e}", exc_info=True)

def display_validation_summary(validation_results: list):
    st.markdown('<h3 class="step-header">Validation Results Summary</h3>', unsafe_allow_html=True)
    summary_df = pd.DataFrame(validation_results)
    
    def style_validation_results(val):
        if isinstance(val, str) and val.endswith('%'):
            percentage = float(val.rstrip('%'))
            color = '#27AE60' if percentage >= VALIDATION_CONFIG["pass_threshold"] else '#E74C3C'
            return f'color: {color}; font-weight: bold'
        return ''
    
    st.dataframe(
        summary_df.style.applymap(style_validation_results),
        use_container_width=True
    )

def display_detailed_validation_results(df: pd.DataFrame, validation_results: list, validation_rules: dict):
    st.subheader("Detailed Validation Results")
    for result in validation_results:
        if result["Fail"] > 0:
            col = result[Column.NAME.value]
            st.write(f"### Column: {col}")
            rule = result["Rule"]
            
            try:
                if rule == "Null values":
                    filtered_df = df[df[col].isnull()]
                elif rule == "Unique values":
                    filtered_df = df[df.duplicated(subset=[col], keep=False)]
                elif rule == "Values outside allowed list":
                    allowed = validation_rules[col][VRule.VALIDATE_LIST_OF_VALUES.value]
                    filtered_df = df[~df[col].isin(allowed)]
                elif rule == "Values not matching regex":
                    regex_pattern = validation_rules[col][VRule.VALIDATE_REGEX.value]
                    filtered_df = df[~df[col].astype(str).str.match(regex_pattern, na=False)]
                elif rule == "Values out of range":
                    min_val = validation_rules[col].get(VRule.MIN_VALUE.value)
                    max_val = validation_rules[col].get(VRule.MAX_VALUE.value)
                    numeric_col = pd.to_numeric(df[col], errors='coerce')
                    # Combine conditions: flag values less than min OR greater than max
                    condition = pd.Series(False, index=numeric_col.index)
                    if min_val is not None:
                        condition |= (numeric_col < float(min_val))
                    if max_val is not None:
                        condition |= (numeric_col > float(max_val))
                    filtered_df = df[condition]
                
                if filtered_df is not None:
                    st.write(f"Failed records count: {len(filtered_df)}")
                    display_df = clean_dataframe_for_display(filtered_df)
                    st.dataframe(display_df, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error displaying validation results for column {col}: {str(e)}")
                logger.error(f"Error in display_detailed_validation_results: {e}", exc_info=True)
            
            st.write("---")

if __name__ == "__main__":
    main()