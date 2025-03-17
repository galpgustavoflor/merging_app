import logging
import time
import streamlit as st
import pandas as pd
import json
import dask.dataframe as dd
from typing import Dict, List, Any

from config import VALIDATION_CONFIG
from constants import Column, Step, ValidationRule as VRule, Functions, FILE_TYPES  # Changed import location
from utils import FileLoader, clean_dataframe_for_display
from state import SessionState

# Import business logic
from logic import (
    MappingProcessor, MatchingProcessor, ValidationProcessor, ReportGenerator, 
    convert_pandas_to_dask_if_needed, ensure_matching_results_are_dask, get_dataframe_sample
)

# Import UI components
from ui import (
    display_business_rules_summary, display_comprehensive_report, display_match_quality_insights, display_validation_summary_charts, inject_custom_css, render_breadcrumbs, step_navigation, page_header, step_header, 
    file_upload_form, mapping_json_uploader, key_selection_form, column_mapping_form, string_transformation_form,
    validation_rules_form, business_rules_form, display_business_rules,
    display_dataframe_metadata, display_matching_results, display_validation_summary,
    display_detailed_validation_results, display_business_rule_violations,
    display_report_summary, soda_config_form, display_soda_yaml,
    display_download_pdf_button, display_download_csv_button, show_spinner_with_message,
    display_mapping_effectiveness  # Explicitly import this function
)

logger = logging.getLogger(__name__)

def main():
    """Main entry point for the Streamlit app."""
    # Get page title from config
    page_title = SessionState.get_config('streamlit_config', 'page_title', "Dataset Comparison and Validation Tool")
    
    st.set_page_config(
        page_title=page_title,
        layout="wide",
        initial_sidebar_state="collapsed",
        page_icon="🍊"
    )
    
    # Set up task state handler instead of trying to register routes
    try:
        from routes import setup_task_state_handler
        setup_task_state_handler()
        logger.info("Task state handler initialized")
    except Exception as e:
        logger.warning(f"Could not initialize task state handler: {e}")
    
    # Use centralized session state initialization
    SessionState.initialize()
    
    inject_custom_css()
    
    with st.container():
        page_header("File Mapping &amp; Validation")
    
    # Render breadcrumbs with callback for navigation
    render_breadcrumbs(
        current_step=SessionState.get_step(), 
        on_step_click=lambda step: handle_step_click(step)
    )
    
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
            Step.REPORT_SUMMARY: handle_report_summary
        }
        
        current_handler = step_handlers.get(Step(SessionState.get_step()))
        if current_handler:
            current_handler()
        st.markdown('</div>', unsafe_allow_html=True)

def handle_step_click(step):
    """Handle click on breadcrumb step"""
    SessionState.go_to_step(step)
    st.rerun()

def handle_source_file_upload():
    step_header("Step 1: Load Source File")
    
    # Get allowed file types from config
    file_types = SessionState.get_config('streamlit_config', 'file_types', FILE_TYPES)
    
    uploaded_source = file_upload_form(
        label="Load Source File", 
        file_types=file_types, 
        key="source_uploader"
    )
    
    if uploaded_source:
        df = FileLoader.load_file(uploaded_source)
        if df is not None:
            # Initially store the raw dataframe
            SessionState.set_dataframe('df_source', df)
            
            # Show string transformation options
            with st.expander("Apply String Transformations", expanded=True):
                transformations = string_transformation_form(df)
                
                if transformations and st.button("Apply Transformations", key="apply_source_transforms"):
                    with st.spinner("Applying transformations..."):
                        # Apply the transformations
                        from utils import apply_string_transformations
                        # Log sample before transformation for debugging
                        if len(df) > 0:
                            sample_before = df.iloc[0].to_dict()
                            logger.debug(f"Sample before transformation: {sample_before}")
                        
                        df = apply_string_transformations(df, transformations)
                        
                        # Log sample after transformation for debugging
                        if len(df) > 0:
                            sample_after = df.iloc[0].to_dict()
                            logger.debug(f"Sample after transformation: {sample_after}")
                        
                        # Store the transformed dataframe, replacing the original
                        SessionState.set_dataframe('df_source', df)
                        # Store applied transformations for reference
                        SessionState.set_value('source_transformations', transformations)
                        # Store the transformed data in a separate key for safekeeping
                        SessionState.set_value('df_source_transformed', df.copy())
                        st.success("Transformations applied successfully!")
            
            # Display the current dataframe (original or transformed)
            df_to_display = SessionState.get_dataframe('df_source')
            display_dataframe_metadata(df_to_display, "Source Data")
            
            step_navigation(
                current_step=SessionState.get_step(),
                on_next=SessionState.next_step,
                back=False
            )

def handle_target_file_upload():
    step_header("Step 2: Load Target File")
    
    # Get allowed file types from config
    file_types = SessionState.get_config('streamlit_config', 'file_types', FILE_TYPES)
    
    uploaded_target = file_upload_form(
        label="Load Target File", 
        file_types=file_types, 
        key="target_uploader"
    )
    
    if uploaded_target:
        df = FileLoader.load_file(uploaded_target)
        if df is not None:
            # Initially store the raw dataframe
            SessionState.set_dataframe('df_target', df)
            
            # Show string transformation options
            with st.expander("Apply String Transformations", expanded=True):
                transformations = string_transformation_form(df)
                
                if transformations and st.button("Apply Transformations", key="apply_target_transforms"):
                    with st.spinner("Applying transformations..."):
                        # Apply the transformations
                        from utils import apply_string_transformations
                        df = apply_string_transformations(df, transformations)
                        # Store the transformed dataframe, replacing the original
                        SessionState.set_dataframe('df_target', df)
                        # Store applied transformations for reference
                        SessionState.set_value('target_transformations', transformations)
                        # Store the transformed data in a separate key for safekeeping
                        SessionState.set_value('df_target_transformed', df.copy())
                        st.success("Transformations applied successfully!")
            
            # Display the current dataframe (original or transformed)
            df_to_display = SessionState.get_dataframe('df_target')
            display_dataframe_metadata(df_to_display, "Target Data")
            
            step_navigation(
                current_step=SessionState.get_step(),
                on_back=SessionState.prev_step,
                on_next=SessionState.next_step
            )

def handle_mapping_rules():
    st.header("Step 3: Define Mapping Rules and Keys")
    
    # Initialize mapping in session state if not exists
    mapping = SessionState.get_value("mapping", {"mappings": {}, "key_source": [], "key_target": []})
    
    # Handle JSON upload with safe state management
    uploaded_json = mapping_json_uploader()
    
    # Store the file upload state
    previous_upload = SessionState.get_value("previous_upload")
        
    # Only process if there's a new file and it's different from the previous one
    if uploaded_json and uploaded_json != previous_upload:
        try:
            mapping_data = json.load(uploaded_json)
            
            # Use MappingProcessor to validate structure
            is_valid, message = MappingProcessor.validate_json_structure(mapping_data)
            
            if is_valid:
                SessionState.update_mapping(mapping_data)
                SessionState.set_value("previous_upload", uploaded_json)
                st.success("Mapping rules loaded successfully!")
            else:
                st.error(f"Invalid mapping file structure: {message}")
        except Exception as e:
            st.error(f"Error loading mapping file: {str(e)}")
    
    # Validate required data
    df_source = SessionState.get_dataframe("df_source")
    df_target = SessionState.get_dataframe("df_target")
    if df_source is None or df_target is None:
        st.error("Please upload both source and target files.")
        return
    
    # Display what transformations were applied to help debugging
    source_transformations = SessionState.get_value("source_transformations", {})
    target_transformations = SessionState.get_value("target_transformations", {})
    
    if source_transformations:
        st.info("Source dataset has the following transformations applied:")
        for col, transforms in source_transformations.items():
            st.write(f"- Column '{col}': {', '.join(transforms)}")
            
    if target_transformations:
        st.info("Target dataset has the following transformations applied:")
        for col, transforms in target_transformations.items():
            st.write(f"- Column '{col}': {', '.join(transforms)}")
    
    # Create a working copy of mapping config
    mapping_config = mapping.copy()
    
    # Key selection with defaults from current mapping
    key_source, key_target = key_selection_form(
        df_source=df_source,
        df_target=df_target,
        default_source_keys=mapping_config.get("key_source", []),
        default_target_keys=mapping_config.get("key_target", [])
    )
    
    # Column mapping configuration
    column_mappings, mapped_columns = column_mapping_form(
        df_source=df_source, 
        df_target=df_target, 
        mapping_config=mapping_config,
        key_source=key_source,
        functions_list=Functions
    )
    
    # Use MappingProcessor to update the config
    updated_config = MappingProcessor.process_mapping_config(
        mapping_config,
        key_source,
        key_target,
        df_source.columns.tolist(),
        mapped_columns,
        column_mappings
    )
    
    # Update session state
    SessionState.update_mapping(updated_config)
    
    # Display current mapping configuration
    if st.checkbox("Show Mapping Configuration", key="show_mapping"):
        st.json(updated_config)
    
    # Navigation
    if key_source and key_target and updated_config.get("mappings"):
        step_navigation(
            current_step=SessionState.get_step(),
            on_back=SessionState.prev_step,
            on_next=SessionState.next_step
        )
    else:
        st.warning("Please configure at least one mapping and select keys before proceeding.")
        step_navigation(
            current_step=SessionState.get_step(),
            on_back=SessionState.prev_step,
            next=False
        )

def handle_matching_execution():
    st.header("Step 4: Execute Matching")
    
    # Get mapping configuration
    mapping_config = SessionState.get_value("mapping")
    if not mapping_config:
        st.error("Please define mapping rules first.")
        step_navigation(
            current_step=SessionState.get_step(),
            on_back=SessionState.prev_step,
            next=False
        )
        return
    
    # Get data from session state - prioritize using the separately stored transformed dataframes if available
    df_source = SessionState.get_value('df_source_transformed', SessionState.get_dataframe("df_source"))
    df_target = SessionState.get_value('df_target_transformed', SessionState.get_dataframe("df_target"))
    
    # Show if transformations have been applied
    source_transformations = SessionState.get_value("source_transformations", {})
    target_transformations = SessionState.get_value("target_transformations", {})
    
    if source_transformations:
        st.info("Source dataset has the following transformations applied:")
        for col, transforms in source_transformations.items():
            st.write(f"- Column '{col}': {', '.join(transforms)}")
            
    if target_transformations:
        st.info("Target dataset has the following transformations applied:")
        for col, transforms in target_transformations.items():
            st.write(f"- Column '{col}': {', '.join(transforms)}")
    
    if df_source is None or df_target is None:
        st.error("Please upload both source and target files.")
        return
    
    try:
        # Check if we already have results or a task is running
        has_matching_results = SessionState.has_value("matching_results")
        task_id = SessionState.get_value("matching_task_id")
        is_task_running = SessionState.get_value("matching_in_progress", False)
        
        # Create containers
        execution_container = st.container()
        progress_container = st.container()
        results_container = st.container()
        
        with execution_container:
            # Show execute button if no task is running and no results exist
            if not has_matching_results and not is_task_running:
                if st.button("Execute Matching", type="primary"):
                    with st.spinner("Initializing matching process..."):
                        # Submit the task
                        from async_utils import async_execute_matching
                        
                        # Display a message while initializing
                        st.info("Preparing to execute matching. This might take a moment...")
                        
                        # Submit task with delay display to ensure UI updates
                        task_id = async_execute_matching(df_source, df_target, mapping_config)
                        
                        # Store task ID and set in-progress status
                        SessionState.set_value("matching_task_id", task_id)
                        SessionState.set_value("matching_in_progress", True)
                        
                        # Force rerun
                        st.rerun()
            
            # Show re-run button if results exist
            elif has_matching_results:
                if st.button("Re-run Matching"):
                    # Clear existing results
                    SessionState.clear_state("matching_results")
                    SessionState.clear_state("matching_stats")
                    SessionState.clear_state("matching_sample")
                    SessionState.clear_state("matching_in_progress")
                    SessionState.clear_state("matching_task_id")
                    st.rerun()
        
        # Show task status if a task is running
        with progress_container:
            if is_task_running and task_id:
                st.subheader("Execution Progress")
                
                # Display debug info in expander
                with st.expander("Debug Info", expanded=False):
                    from async_utils import TaskManager
                    task_manager = TaskManager()
                    task_status = task_manager.get_task_status(task_id)
                    st.write("Task ID:", task_id)
                    st.write("Task Status:", task_status)
                
                try:
                    # Use standard progress display for reliable updates
                    from async_utils import TaskManager, TaskStatus
                    
                    # Get task status
                    task_manager = TaskManager()
                    task_status = task_manager.get_task_status(task_id)
                    status = task_status["status"]
                    progress = task_status["progress"] or 0.0
                    
                    # Display progress bar with custom message based on progress level
                    progress_message = "Starting matching process..."
                    if progress >= 0.2 and progress < 0.6:
                        progress_message = "Matching records..."
                    elif progress >= 0.6 and progress < 0.8:
                        progress_message = "Processing results..."
                    elif progress >= 0.8:
                        progress_message = "Preparing sample data..."
                        
                    st.progress(progress, text=f"{progress_message} ({int(progress * 100)}%)")
                    
                    # Show error if failed
                    if status == TaskStatus.FAILED.value:
                        st.error(f"Task failed: {task_status.get('error', 'Unknown error')}")
                        SessionState.set_value("matching_in_progress", False)
                    
                    # Check if task completed
                    elif status == TaskStatus.COMPLETED.value:
                        st.success("Task completed! Processing results...")
                        
                        # Get result and store in session state
                        result = task_manager.get_task_result(task_id)
                        if result is not None:
                            # Unpack results
                            ddf_merged, stats, sample_df = result
                            
                            # Update session state
                            SessionState.set_value("matching_results", ddf_merged)
                            SessionState.set_value("matching_stats", stats)
                            SessionState.set_value("matching_sample", sample_df)
                            SessionState.set_value("matching_in_progress", False)
                            
                            # Show success message and rerun to update UI
                            st.rerun()
                    
                    # If still running, auto-refresh
                    elif status == TaskStatus.RUNNING.value or status == TaskStatus.PENDING.value:
                        # Use a hidden trigger for rerun that won't affect the visible UI
                        time.sleep(1)  # Brief pause to avoid hammering the server
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error displaying task status: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Show results if available
        with results_container:
            if SessionState.has_value("matching_stats") and SessionState.has_value("matching_sample"):
                # Get the results from session state
                stats = SessionState.get_value("matching_stats") 
                sample_df = SessionState.get_value("matching_sample")
                
                # Display matching results
                display_matching_results(stats, sample_df)
    
    except Exception as e:
        st.error(f"Error during matching execution: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        logger.error(f"Matching execution error: {e}", exc_info=True)
    
    step_navigation(
        current_step=SessionState.get_step(),
        on_back=SessionState.prev_step,
        on_next=SessionState.next_step,
        next_label="Configure Validations"
    )

def handle_validation_rules():
    st.header("Step 5: Define Validation Rules")
    
    # Initial checks - Use proper has_value check instead of direct boolean evaluation
    if not SessionState.has_value("matching_results"):
        st.error("Please execute matching first")
        step_navigation(
            current_step=SessionState.get_step(),
            on_back=SessionState.prev_step,
            next=False
        )
        return
    
    df_target = SessionState.get_dataframe("df_target")
    if df_target is None:
        st.error("Target dataset not found")
        return

    # Initialize columns in session state if needed
    columns = SessionState.get_value("columns", [])
    if not columns and df_target is not None:
        columns = df_target.columns.tolist()
        SessionState.set_value("columns", columns)
    
    if not columns:
        st.error("No columns available for validation")
        return

    # Create tabs for data quality rules and business rules
    main_tabs = st.tabs(["📊 Data Quality Rules", "🔍 Business Rules"])
    
    with main_tabs[0]:
        # Data Quality Rules section
        st.subheader("Data Quality Rules")
        
        # Get or initialize validation rules
        validation_rules = SessionState.get_value("validation_rules", {})
        
        # Display the validation rules form
        updated_rules = validation_rules_form(df_target, validation_rules)
        
        # Update session state if rules have changed
        if updated_rules != validation_rules:
            SessionState.update_validation_rules(updated_rules)
    
    with main_tabs[1]:
        # Business Rules section
        st.subheader("Business Rules")
        
        # Get or initialize business rules
        business_rules = SessionState.get_value("business_rules", [])
        
        # Handle rule creation/deletion
        rule_dialog = st.session_state.get("show_rule_dialog", False)
        current_rule = st.session_state.get("current_rule", {"name": "", "conditions": [], "then": []})
        rule_name = st.session_state.get("rule_name", "")
        
        # Define callbacks
        def on_save_rule(name):
            # Create a copy of the rule with the new name
            rule = current_rule.copy()
            rule["name"] = name
            
            # Add to business rules
            business_rules.append(rule)
            
            # Reset state
            st.session_state.show_rule_dialog = False
            st.session_state.current_rule = {"name": "", "conditions": [], "then": []}
            st.session_state.rule_name = ""
            
            # Store updated rules
            SessionState.set_value("business_rules", business_rules)
            st.rerun()
        
        def on_cancel_rule():
            # Reset state
            st.session_state.show_rule_dialog = False
            st.session_state.current_rule = {"name": "", "conditions": [], "then": []}
            st.session_state.rule_name = ""
            st.rerun()
        
        def on_add_condition(condition):
            # Add to current rule
            current_rule["conditions"].append(condition)
            st.session_state.current_rule = current_rule
            st.rerun()
        
        def on_add_then(then_condition):
            # Add to current rule
            current_rule["then"].append(then_condition)
            st.session_state.current_rule = current_rule
            st.rerun()
        
        def on_delete_rule(index):
            # Remove rule at index
            del business_rules[index]
            SessionState.set_value("business_rules", business_rules)
            st.rerun()
        
        # Display create rule button
        if not rule_dialog and st.button("Create New Rule"):
            st.session_state.show_rule_dialog = True
            st.rerun()
        
        # Display rule creation form
        business_rules_form(
            columns=columns,
            current_rule=current_rule,
            show_dialog=rule_dialog,
            rule_name=rule_name,
            on_save=on_save_rule,
            on_cancel=on_cancel_rule,
            on_add_condition=on_add_condition,
            on_add_then=on_add_then
        )
        
        # Display existing rules
        from utils import format_rule_as_sentence  # Local import to avoid circular dependencies
        
        display_business_rules(
            business_rules=business_rules,
            format_rule_func=format_rule_as_sentence,
            on_delete=on_delete_rule
        )
    
    step_navigation(
        current_step=SessionState.get_step(),
        on_back=SessionState.prev_step,
        on_next=SessionState.next_step
    )

def handle_data_validation():
    st.header("Step 6: Execute Data Validation")
    
    # Check if we have any validation rules or business rules
    validation_rules = SessionState.get_value("validation_rules", {})
    business_rules = SessionState.get_value("business_rules", [])
    
    if not validation_rules and not business_rules:
        st.error("Please define at least one validation rule or business rule before proceeding.")
        step_navigation(
            current_step=SessionState.get_step(),
            on_back=SessionState.prev_step,
            next=False
        )
        return
    
    df_target = SessionState.get_dataframe("df_target")
    if df_target is None:
        st.error("Target data is required for validation.")
        step_navigation(
            current_step=SessionState.get_step(),
            on_back=SessionState.prev_step,
            next=False
        )
        return
    
    # Execute standard validation rules if they exist
    validation_results = []
    if validation_rules:
        st.subheader("Standard Validation Rules")
        
        # Use ValidationProcessor instead of DataValidator directly
        with st.spinner("Executing standard validation rules..."):
            validation_results = ValidationProcessor.execute_standard_validation(
                df_target, validation_rules
            )
            SessionState.set_value("validation_results", validation_results)
        
        display_validation_summary(validation_results)
        display_detailed_validation_results(df_target, validation_results, validation_rules)
    
    # Execute business rules validation if they exist
    business_rule_violations = {}
    if business_rules:
        st.subheader("Business Rules Validation")
        
        # Use ValidationProcessor for business rules validation
        with show_spinner_with_message("Validating business rules..."):
            business_rule_violations = ValidationProcessor.execute_business_rules(
                df_target, business_rules
            )
            
        if business_rule_violations:
            st.error("Business rules violations found:")
            display_business_rule_violations(
                df_target, 
                business_rule_violations, 
                business_rules, 
                ValidationProcessor.format_rule_as_sentence
            )
        else:
            st.success("All business rules passed!")

    # Store validation results for final report
    SessionState.set_value("business_rule_violations", business_rule_violations)
    
    # Only show next step if we have results
    if validation_rules or (business_rules and len(business_rules) > 0):
        step_navigation(
            current_step=SessionState.get_step(),
            on_back=SessionState.prev_step,
            on_next=SessionState.next_step
        )
    else:
        step_navigation(
            current_step=SessionState.get_step(),
            on_back=SessionState.prev_step,
            next=False
        )

def handle_report_summary():
    """Generate and display final comprehensive report."""
    step_header("Step 7: Final Report & Analysis")
    
    try:
        # Get the necessary data from session state - using cached values
        source_df = SessionState.get_dataframe("df_source")
        target_df = SessionState.get_dataframe("df_target")
        matching_results = SessionState.get_value("matching_results")
        mapping_config = SessionState.get_value("mapping", {})
        validation_results = SessionState.get_value("validation_results", [])
        business_rules = SessionState.get_value("business_rules", [])
        business_rule_violations = SessionState.get_value("business_rule_violations", {})
        
        # Get matching statistics
        matching_stats = SessionState.get_value("matching_stats", {})
        
        # Generate comprehensive report data
        with st.spinner("Generating comprehensive report..."):
            report_data = ReportGenerator.generate_comprehensive_report(
                source_df,
                target_df,
                matching_results,
                mapping_config,
                matching_stats,
                validation_results,
                business_rules,
                business_rule_violations
            )
            
            # Store the report data in session state for potential reuse
            SessionState.set_value("comprehensive_report", report_data)
            
            # Display the comprehensive report
            display_comprehensive_report(report_data)
            
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {str(e)}", exc_info=True)
        st.error(f"An error occurred while generating the report: {str(e)}")
        
        # Fallback to simple summary if comprehensive report fails
        st.markdown("### Fallback Report Summary")
        st.warning("The comprehensive report could not be generated. Here's a simplified summary:")
        
        # Display basic metrics if available
        source_summary = SessionState.get_value("source_summary", {})
        target_summary = SessionState.get_value("target_summary", {})
        matching_count = SessionState.get_value("matching_count", 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Source Records", source_summary.get("rows", "N/A"))
            st.metric("Target Records", target_summary.get("rows", "N/A"))
        with col2:
            st.metric("Matched Records", matching_count)
    
    # Navigation buttons
    step_navigation(
        SessionState.get_step(),
        on_back=SessionState.prev_step,
        next=False
    )

# Helper function to display executive summary with caching
@st.cache_data(ttl=600)
def display_executive_summary(source_summary, target_summary, matching_stats, 
                           validation_results, business_rules, business_rule_violations):
    st.subheader("Executive Summary")
    
    # Create 3 columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Source Records", 
            f"{source_summary.get('rows', 0):,}", 
            help="Total number of records in source dataset"
        )
    with col2:
        st.metric(
            "Target Records", 
            f"{target_summary.get('rows', 0):,}",
            help="Total number of records in target dataset"
        )
    with col3:
        match_pct = matching_stats.get('match_percentage', 0)
        st.metric(
            "Match Rate", 
            f"{match_pct:.2f}%",
            delta=f"{match_pct - 100:.2f}%" if match_pct < 100 else None,
            delta_color="inverse",
            help="Percentage of records that matched between source and target"
        )
    
    st.markdown("---")
    
    # Data Quality Summary - only compute when needed
    if validation_results:
        display_validation_summary_charts(validation_results)
            
    # Business Rules Summary - only compute when needed
    if business_rules:
        display_business_rules_summary(business_rules, business_rule_violations)

if __name__ == "__main__":
    main()
