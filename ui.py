import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import logging
from typing import Dict, List, Any, Optional, Union, Callable
from constants import Column, ValidationRule as VRule, Step, STEP_LABELS, COMPARISON_OPERATORS

# Set up module logger
logger = logging.getLogger(__name__)

def inject_custom_css():
    """Inject custom CSS for modern styling."""
    with open('styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# -------------------------
# Navigation UI Components
# -------------------------

def render_breadcrumbs(current_step: int, on_step_click: Callable):
    """Render breadcrumb navigation showing progress through application steps.
    
    Args:
        current_step: Current step value
        on_step_click: Callback function when step is clicked
    """
    with st.container():
        st.markdown('<div class="breadcrumb-container" style="display: flex; justify-content: center; gap: 1em; margin-bottom: 1em;">', unsafe_allow_html=True)
        cols = st.columns(len(STEP_LABELS))
        for i, (step, label) in enumerate(STEP_LABELS.items()):
            with cols[i]:
                if step.value < current_step:
                    if st.button(f"✔ {label}", key=f"bread_{step}", help="Go to this step"):
                        on_step_click(step.value)
                elif step.value == current_step:
                    st.markdown(f'<div class="breadcrumb-active" style="color: #FF6600; font-weight: bold;">● {label}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="breadcrumb-pending" style="color: #B0BEC5;">○ {label}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def step_navigation(current_step: int, on_back: Callable = None, on_next: Callable = None, 
                   back: bool = True, next: bool = True, next_label: str = "Next Step"):
    """Render navigation buttons for moving between steps.
    
    Args:
        current_step: Current step value
        on_back: Callback function for back button
        on_next: Callback function for next button
        back: Show back button
        next: Show next button
        next_label: Label for next button
    """
    cols = st.columns(2)
    if back and current_step > Step.SOURCE_UPLOAD.value and on_back:
        if cols[0].button("← Back", key="back_button", help="Go to previous step", use_container_width=True):
            on_back()
            st.rerun()  # Add rerun to immediately apply the state change
    if next and on_next:
        if cols[1].button(f"{next_label} →", key="next_button", help="Proceed to next step", use_container_width=True):
            on_next()
            st.rerun()  # Add rerun to immediately apply the state change

def page_header(title: str):
    """Display page header with title.
    
    Args:
        title: Page title
    """
    st.markdown(
        f'<h1 class="main-title" style="text-align: center; font-size: 3em; margin-bottom: 0.5em; color: #FF6600; font-family: \'Segoe UI\', Tahoma, Geneva, Verdana, sans-serif;">'
        f'{title}'
        '</h1>',
        unsafe_allow_html=True
    )

def step_header(title: str):
    """Display step header.
    
    Args:
        title: Step title
    """
    st.markdown(f'<h2 class="step-header">{title}</h2>', unsafe_allow_html=True)

# -------------------------
# Form Elements
# -------------------------

def file_upload_form(label: str, file_types: List[str], key: str, help_text: str = None):
    """Generic file upload form.
    
    Args:
        label: Upload button label
        file_types: Allowed file types
        key: Unique key for component
        help_text: Optional help text
        
    Returns:
        The uploaded file object or None
    """
    return st.file_uploader(
        label, 
        type=file_types, 
        key=key,
        help=help_text
    )

def mapping_json_uploader():
    """File uploader for JSON mapping files.
    
    Returns:
        Uploaded JSON file object or None
    """
    with st.expander("Upload JSON File for Mapping"):
        return st.file_uploader(
            "Load Mapping Rules from JSON", 
            type=["json"], 
            key="mapping_json",
            help="Upload a JSON file containing mapping rules"
        )

def key_selection_form(df_source: pd.DataFrame, df_target: pd.DataFrame, 
                     default_source_keys: List[str], default_target_keys: List[str]):
    """Form for selecting source and target keys.
    
    Args:
        df_source: Source dataframe
        df_target: Target dataframe
        default_source_keys: Default selected source keys
        default_target_keys: Default selected target keys
        
    Returns:
        Tuple of (selected source keys, selected target keys)
    """
    col1, col2 = st.columns(2)
    with col1:
        key_source = st.multiselect(
            "Select source key(s)", 
            df_source.columns.tolist(),
            default=default_source_keys,
            key="key_source_select"
        )
    with col2:
        key_target = st.multiselect(
            "Select target key(s)",
            df_target.columns.tolist(),
            default=default_target_keys,
            key="key_target_select"
        )
    return key_source, key_target

def column_mapping_form(df_source: pd.DataFrame, df_target: pd.DataFrame, 
                      mapping_config: Dict, key_source: List[str], functions_list: List):
    """Form for mapping source columns to target columns."""
    st.subheader("Column Mappings")
    mapped_columns = set()
    column_mappings = {}
    
    # Get the first function value as default
    function_values = [f.value for f in functions_list]
    default_function = function_values[0] if function_values else "Direct mapping"
    
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
                    
                    # Mapping function selection - fixed indexing issue
                    current_function = col_config.get("function", default_function)
                    try:
                        function_index = function_values.index(current_function)
                    except ValueError:
                        function_index = 0
                        
                    function = st.selectbox(
                        "Mapping Type",
                        function_values,
                        index=function_index,
                        key=f"function_select_{col}"
                    )
                    
                    # Handle transformation based on function
                    transformation = handle_transformation_ui(
                        col, function, col_config, df_source
                    )
                    
                    # Store mapping configuration for this column
                    if mapped_cols:
                        column_mappings[col] = {
                            "destinations": mapped_cols,
                            "function": function,
                            "transformation": transformation
                        }
                        mapped_columns.add(col)
                        
    return column_mappings, list(mapped_columns)

def handle_transformation_ui(col, function, col_config, df_source):
    """UI for handling column transformations based on function type.
    
    Args:
        col: Column name
        function: Selected function type
        col_config: Current column configuration
        df_source: Source dataframe
        
    Returns:
        Transformation configuration
    """
    from logic import MappingProcessor  # Import here to avoid circular imports
    
    if function == "Conversion mapping":
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
            
            # Validate JSON using MappingProcessor
            is_valid, message, transformation_dict = MappingProcessor.parse_and_validate_transformation(
                transformation, function
            )
            
            if not is_valid:
                st.error(message)
                transformation = "{}"
        
        with conv_col2:
            # Visual mapping table
            st.write("Visual Mapping Table")
            
            # Get unique values from source column
            unique_values = df_source[col].unique()
            
            # Create mapping table
            mapping_data = []
            try:
                import json
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
            
        return transformation
    
    elif function == "Aggregation":
        return st.selectbox(
            "Aggregation Function",
            ["sum", "mean", "median", "max", "min"],
            index=0,
            key=f"agg_select_{col}"
        )
    
    return None

def validation_rules_form(df: pd.DataFrame, validation_rules: Dict):
    """Form for defining validation rules.
    
    Args:
        df: Target dataframe to validate
        validation_rules: Current validation rules
        
    Returns:
        Updated validation rules
    """
    # Create a deep copy to avoid modifying the input dictionary
    rules = validation_rules.copy()
    
    # Group columns by data type
    numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    text_cols = [col for col in df.columns if col not in numeric_cols]
    
    # Create sections for different column types
    if numeric_cols:
        st.subheader("Numeric Columns")
        for col in numeric_cols:
            with st.expander(f"📊 {col}", expanded=col in rules):
                col1, col2 = st.columns(2)
                with col1:
                    if st.checkbox("Check Nulls", key=f"nulls_{col}", 
                                 value=rules.get(col, {}).get(VRule.VALIDATE_NULLS.value, False)):
                        rules.setdefault(col, {})[VRule.VALIDATE_NULLS.value] = True
                    
                    if st.checkbox("Check Uniqueness", key=f"unique_{col}", 
                                 value=rules.get(col, {}).get(VRule.VALIDATE_UNIQUENESS.value, False)):
                        rules.setdefault(col, {})[VRule.VALIDATE_UNIQUENESS.value] = True
                
                with col2:
                    if st.checkbox("Check Range", key=f"range_{col}", 
                                 value=rules.get(col, {}).get(VRule.VALIDATE_RANGE.value, False)):
                        rules.setdefault(col, {})[VRule.VALIDATE_RANGE.value] = True
                        min_col, max_col = st.columns(2)
                        with min_col:
                            if st.checkbox("Set Min", key=f"has_min_{col}",
                                         value=rules.get(col, {}).get(VRule.MIN_VALUE.value) is not None):
                                min_value = st.number_input(
                                    "Min Value",
                                    key=f"min_{col}",
                                    value=float(rules.get(col, {}).get(VRule.MIN_VALUE.value, 0.0))
                                )
                                rules[col][VRule.MIN_VALUE.value] = min_value
                        with max_col:
                            if st.checkbox("Set Max", key=f"has_max_{col}",
                                         value=rules.get(col, {}).get(VRule.MAX_VALUE.value) is not None):
                                max_value = st.number_input(
                                    "Max Value",
                                    key=f"max_{col}",
                                    value=float(rules.get(col, {}).get(VRule.MAX_VALUE.value, 0.0))
                                )
                                rules[col][VRule.MAX_VALUE.value] = max_value
    
    if text_cols:
        st.subheader("Text Columns")
        for col in text_cols:
            with st.expander(f"📝 {col}", expanded=col in rules):
                col1, col2 = st.columns(2)
                with col1:
                    if st.checkbox("Check Nulls", key=f"nulls_{col}", 
                                 value=rules.get(col, {}).get(VRule.VALIDATE_NULLS.value, False)):
                        rules.setdefault(col, {})[VRule.VALIDATE_NULLS.value] = True
                    
                    if st.checkbox("Check Uniqueness", key=f"unique_{col}", 
                                 value=rules.get(col, {}).get(VRule.VALIDATE_UNIQUENESS.value, False)):
                        rules.setdefault(col, {})[VRule.VALIDATE_UNIQUENESS.value] = True
                
                with col2:
                    # Allowed values section
                    if st.checkbox("Check Allowed Values", key=f"domain_{col}"):
                        values = st.text_input(
                            "Allowed Values (comma separated)",
                            value=",".join(rules.get(col, {}).get(VRule.VALIDATE_LIST_OF_VALUES.value, [])),
                            key=f"domain_values_{col}"
                        )
                        rules.setdefault(col, {})[VRule.VALIDATE_LIST_OF_VALUES.value] = \
                            [val.strip() for val in values.split(',') if val.strip()]
                    
                    # Regex pattern section
                    if st.checkbox("Check Format (Regex)", key=f"regex_{col}"):
                        pattern = st.text_input(
                            "Regex Pattern",
                            value=rules.get(col, {}).get(VRule.VALIDATE_REGEX.value, ""),
                            key=f"regex_pattern_{col}"
                        )
                        rules.setdefault(col, {})[VRule.VALIDATE_REGEX.value] = pattern
    
    return rules

def business_rules_form(columns: List[str], current_rule: Dict, show_dialog: bool, rule_name: str, 
                      on_save: Callable, on_cancel: Callable, on_add_condition: Callable, 
                      on_add_then: Callable):
    """Form for creating business rules.
    
    Args:
        columns: Available columns list
        current_rule: Current rule being edited
        show_dialog: Whether to show rule dialog
        rule_name: Current rule name
        on_save: Callback when rule is saved
        on_cancel: Callback when rule creation is canceled
        on_add_condition: Callback when condition is added
        on_add_then: Callback when then clause is added
        
    Returns:
        Updated rule data
    """
    if not show_dialog:
        return None
    
    # Create a modal-like container with styling
    st.markdown("""
        <style>
            .modal-container {
                background-color: var(--background-color);
                border: 1px solid var(--primary-color);
                border-radius: 0.5rem;
                padding: 2rem;
                margin: 1rem 0;
                position: relative;
            }
            .modal-header {
                margin-bottom: 1.5rem;
                padding-bottom: 1rem;
                border-bottom: 1px solid rgba(250, 250, 250, 0.2);
            }
        </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="modal-container">', unsafe_allow_html=True)
        st.markdown('<div class="modal-header">', unsafe_allow_html=True)
        st.markdown("### Create Business Rule")
        st.markdown('</div>', unsafe_allow_html=True)

        # Rule name with help text
        new_rule_name = st.text_input("Rule Name", value=rule_name,
                    help="Enter a descriptive name for your rule")

        # IF section
        st.subheader("IF Conditions")
        st.markdown("*Define when this rule should apply*")
        cols = st.columns([2, 2, 1, 2])
        
        with cols[0]:
            column = st.selectbox("Column", columns, key="if_column")
        with cols[1]:
            operator = st.selectbox("Operator", list(COMPARISON_OPERATORS.keys()), 
                                  key="if_operator",
                                  help="Choose how to compare values")
        with cols[2]:
            value_type = st.selectbox("Type", ["Value", "Column"], key="if_value_type")
        with cols[3]:
            if value_type == "Column":
                value = st.selectbox("Compare with column", columns, key="if_value_col")
            else:
                value = st.text_input("Value", key="if_value_input")
        
        if st.button("Add IF Condition"):
            if all([column, operator, value]):
                new_condition = {
                    'column': column,
                    'operator': operator,
                    'value': value,
                    'value_type': value_type.lower()
                }
                on_add_condition(new_condition)

        # Display current conditions
        if current_rule.get('conditions'):
            st.markdown("**Current Conditions:**")
            for i, cond in enumerate(current_rule['conditions']):
                st.info(f"{i+1}. {cond['column']} {cond['operator']} {cond['value']}")
        
        # THEN section
        st.subheader("THEN Conditions")
        st.markdown("*Define what should be true when conditions are met*")
        cols = st.columns([2, 2, 1, 2])
        with cols[0]:
            then_column = st.selectbox("Column", columns, key="then_column")
        with cols[1]:
            then_operator = st.selectbox("Operator", list(COMPARISON_OPERATORS.keys()), 
                                  key="then_operator",
                                  help="Choose how to compare values")
        with cols[2]:
            then_value_type = st.selectbox("Type", ["Value", "Column"], key="then_value_type")
        with cols[3]:
            if then_value_type == "Column":
                then_value = st.selectbox("Compare with column", columns, key="then_value_col")
            else:
                then_value = st.text_input("Value", key="then_value_input")
        
        if st.button("Add THEN Condition"):
            if all([then_column, then_operator, then_value]):
                new_then = {
                    'column': then_column,
                    'operator': then_operator,
                    'value': then_value,
                    'value_type': then_value_type.lower()
                }
                on_add_then(new_then)

        # Display current then conditions
        if current_rule.get('then'):
            st.markdown("**Current THEN Conditions:**")
            for i, cond in enumerate(current_rule['then']):
                st.info(f"{i+1}. {cond['column']} {cond['operator']} {cond['value']}")

        # Action buttons at the bottom
        col1, col2, col3 = st.columns([6, 3, 3])
        with col2:
            if st.button("Save Rule", type="primary", use_container_width=True):
                if new_rule_name and current_rule.get('conditions') and current_rule.get('then'):
                    on_save(new_rule_name)
                else:
                    st.error("Please fill all required fields")
        
        with col3:
            if st.button("Cancel", type="secondary", use_container_width=True):
                on_cancel()

        st.markdown('</div>', unsafe_allow_html=True)

def display_business_rules(business_rules: List[Dict], format_rule_func: Callable, on_delete: Callable):
    """Display existing business rules.
    
    Args:
        business_rules: List of business rules
        format_rule_func: Function to format rule as readable sentence
        on_delete: Callback when rule is deleted
    """
    if not business_rules:
        st.info("No business rules defined yet. Click 'Create New Rule' to get started!")
        return
        
    for i, rule in enumerate(business_rules):
        with st.container():
            st.markdown('<div class="validation-card">', unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 8, 1])
            
            col1.markdown(f"**{i+1}.**")
            with col2:
                st.markdown(f"**{rule['name']}**")
                st.info(format_rule_func(rule))
                with st.expander("View Details"):
                    st.json(rule)
            
            if col3.button("🗑️", key=f"delete_rule_{i}", help="Delete rule"):
                on_delete(i)
                
            st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Data Display Components
# -------------------------

def display_dataframe_metadata(df: pd.DataFrame, title: str):
    """Display dataframe metadata with tabs.
    
    Args:
        df: Dataframe to display
        title: Title to show
    """
    from utils import clean_dataframe_for_display
    
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
            # Sort columns for summary
            summary_df = clean_dataframe_for_display(df.describe(include='all'))
            summary_df = summary_df.reindex(sorted(summary_df.columns), axis=1)
            
            # Add string length statistics for string columns
            string_stats = {}
            for col in df.columns:
                if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                    # Calculate string lengths, handling NaNs
                    lengths = df[col].dropna().astype(str).str.len()
                    if not lengths.empty:
                        string_stats[col] = {
                            'min_length': lengths.min(),
                            'max_length': lengths.max(),
                            'mean_length': round(lengths.mean(), 2)
                        }
            
            # Display summary DataFrame
            st.dataframe(summary_df, use_container_width=True)
            
            # Display string length statistics if available
            if string_stats:
                st.subheader("String Length Statistics")
                string_stats_df = pd.DataFrame.from_dict(
                    {col: stats for col, stats in string_stats.items()},
                    orient='index'
                )
                st.dataframe(string_stats_df)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[2]:
            st.markdown('<div class="data-preview">', unsafe_allow_html=True)
            # Data type classification
            dtype_info = []
            for col in display_df.columns:
                col_type = display_df[col].dtype
                
                # Determine data category
                if pd.api.types.is_numeric_dtype(col_type):
                    category = "Numeric"
                    # Further classify numeric data
                    if pd.api.types.is_integer_dtype(col_type):
                        category = "Integer"
                    elif pd.api.types.is_float_dtype(col_type):
                        category = "Float"
                elif pd.api.types.is_datetime64_dtype(col_type):
                    category = "DateTime"
                elif pd.api.types.is_bool_dtype(col_type):
                    category = "Boolean"
                else:
                    category = "Text"
                    # Attempt to detect specific text patterns
                    if not df[col].dropna().empty:
                        sample = df[col].dropna().iloc[0]
                        if isinstance(sample, str):
                            # Check for common patterns
                            if sample.count("-") == 2 and len(sample.split("-")) == 3:
                                category = "Text (Possible Date Format)"
                            elif "@" in sample and "." in sample:
                                category = "Text (Possible Email)"
                            elif sample.isdigit():
                                category = "Text (Numeric string)"
                
                # Generate value mask for sample
                sample_value = str(display_df[col].dropna().iloc[0]) if not display_df[col].dropna().empty else ""
                mask = ""
                if sample_value:
                    for char in sample_value:
                        if char.isalpha():
                            mask += "A"
                        elif char.isdigit():
                            mask += "9"
                        else:
                            mask += char
                
                dtype_info.append({
                    "Column": col,
                    "Type": str(col_type),
                    "Category": category,
                    "Sample": sample_value[:20] + ("..." if len(sample_value) > 20 else ""),
                    "Pattern Mask": mask[:20] + ("..." if len(mask) > 20 else "")
                })
            
            # Display enhanced datatype information
            dtype_df = pd.DataFrame(dtype_info)
            st.dataframe(dtype_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tabs[3]:
            # Sort columns for null analysis
            null_counts = display_df.isnull().sum().sort_index(ascending=False).reset_index()
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

def display_matching_results(stats: Dict, sample_df: pd.DataFrame):
    """Display matching results summary and samples with optimized performance."""
    if not isinstance(stats, dict) or not isinstance(sample_df, pd.DataFrame):
        st.error(f"Invalid input types: stats={type(stats)}, sample_df={type(sample_df)}")
        # Add debugging info to help diagnose issues
        if st.checkbox("Show Debug Info"):
            st.write("Stats content:", stats)
            if isinstance(sample_df, pd.DataFrame):
                st.write("Sample DataFrame columns:", sample_df.columns.tolist())
            else:
                st.write("Sample content is not a DataFrame:", sample_df)
        return
    
    # Check for required keys in stats
    required_keys = ['total_match', 'missing_source', 'missing_target']
    missing_keys = [key for key in required_keys if key not in stats]
    if missing_keys:
        st.error(f"Missing required stats keys: {missing_keys}")
        if st.checkbox("Show Debug Info"):
            st.write("Available keys:", list(stats.keys()))
            st.write("Stats content:", stats)
        return

    st.subheader("Matching Summary")
    
    # Calculate percentages for pie chart - Fix potential Series issue by converting to scalar values
    try:
        # Convert potential Series or NumPy values to Python scalars
        total_match = float(stats['total_match'])
        missing_source = float(stats['missing_source'])
        missing_target = float(stats['missing_target'])
        total_records = total_match + missing_source + missing_target
    except (TypeError, ValueError) as e:
        st.error(f"Error calculating total records: {str(e)}")
        st.write("Stats content:", stats)
        return
    
    if total_records > 0:
        match_percent = round((total_match / total_records) * 100, 2)
        missing_source_percent = round((missing_source / total_records) * 100, 2)
        missing_target_percent = round((missing_target / total_records) * 100, 2)
        
        # Create columns for stats and chart
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Display formatted statistics
            st.metric("Total matched records", f"{int(total_match)} ({match_percent}%)")
            st.metric("Missing in source", f"{int(missing_source)} ({missing_source_percent}%)")
            st.metric("Missing in target", f"{int(missing_target)} ({missing_target_percent}%)")
        
        with col2:
            # Create pie chart
            labels = ['Matched', 'Missing in Source', 'Missing in Target']
            values = [total_match, missing_source, missing_target]
            colors = ['#27AE60', '#F39C12', '#E74C3C']
            
            # Only show chart if there's data
            if any(values):
                import plotly.graph_objects as go
                fig = go.Figure(data=[go.Pie(
                    labels=labels,
                    values=values,
                    marker=dict(colors=colors),
                    hole=.3,
                    textinfo='label+percent'
                )])
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=250)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No records processed or all counts are zero.")

    # Verify the '_merge' column exists
    if '_merge' not in sample_df.columns:
        st.warning("'_merge' column not found in the sample data. Cannot categorize results.")
        if st.checkbox("Show Raw Sample"):
            st.dataframe(sample_df.head(100))
        return
        
    # Simple approach with direct filtering - no caching or complex state management
    tabs = st.tabs(["Matched Records", "Missing in Source", "Missing in Target"])
    
    # Tab 1: Matched records
    with tabs[0]:
        matched_df = sample_df[sample_df['_merge'] == 'both']
        if not matched_df.empty:
            st.write(f"Found {len(matched_df)} matched records in sample")
            st.dataframe(matched_df)
        else:
            st.info("No matched records found in sample")
            
    # Tab 2: Missing in source
    with tabs[1]:
        missing_source_df = sample_df[sample_df['_merge'] == 'right_only']
        if not missing_source_df.empty:
            st.write(f"Found {len(missing_source_df)} records missing in source")
            st.dataframe(missing_source_df)
        else:
            st.info("No records missing in source found in sample")
            
    # Tab 3: Missing in target
    with tabs[2]:
        missing_target_df = sample_df[sample_df['_merge'] == 'left_only']
        if not missing_target_df.empty:
            st.write(f"Found {len(missing_target_df)} records missing in target")
            st.dataframe(missing_target_df)
        else:
            st.info("No records missing in target found in sample")
    
    # Display mapping effectiveness if available
    st.markdown("---")
    
    if 'mapping_comparison' in stats and stats['mapping_comparison']:
        display_mapping_effectiveness(stats['mapping_comparison'])
    else:
        st.info("No mapping comparison data available.")
    
    # Add enhanced matching analysis section
    if total_records > 0 and st.checkbox("Show Enhanced Matching Analysis", value=True):
        st.subheader("Enhanced Matching Analysis")
        
        # Add match rate trend visualization
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create a metrics comparison table
            metrics_data = {
                "Metric": ["Match Rate", "Missing Source Rate", "Missing Target Rate"],
                "Value": [
                    f"{match_percent:.2f}%", 
                    f"{missing_source_percent:.2f}%", 
                    f"{missing_target_percent:.2f}%"
                ],
                "Count": [
                    f"{stats['total_match']:,}", 
                    f"{stats['missing_source']:,}", 
                    f"{stats['missing_target']:,}"
                ]
            }
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
        with col2:
            # Display a more informative metrics distribution
            fig = px.bar(
                x=["Matched", "Missing in Source", "Missing in Target"],
                y=[match_percent, missing_source_percent, missing_target_percent],
                color=["#27AE60", "#F39C12", "#E74C3C"],
                labels={"x": "Category", "y": "Percentage (%)"},
                title="Matching Distribution"
            )
            fig.update_layout(showlegend=False, height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Add match quality analysis if available
        if "_similarity" in sample_df.columns:
            st.subheader("Match Quality Analysis")
            
            # Calculate similarity statistics
            similarity_stats = {
                "min": sample_df["_similarity"].min(),
                "max": sample_df["_similarity"].max(),
                "mean": sample_df["_similarity"].mean(),
                "median": sample_df["_similarity"].median()
            }
            
            # Display similarity metrics
            cols = st.columns(4)
            cols[0].metric("Min Similarity", f"{similarity_stats['min']:.2f}")
            cols[1].metric("Max Similarity", f"{similarity_stats['max']:.2f}")
            cols[2].metric("Average Similarity", f"{similarity_stats['mean']:.2f}")
            cols[3].metric("Median Similarity", f"{similarity_stats['median']:.2f}")
            
            # Show similarity distribution histogram
            if len(sample_df) > 0:
                fig = px.histogram(
                    sample_df, 
                    x="_similarity",
                    nbins=20,
                    title="Similarity Score Distribution",
                    labels={"_similarity": "Similarity Score", "count": "Number of Records"}
                )
                st.plotly_chart(fig, use_container_width=True)

def display_cached_filtered_records(df: pd.DataFrame, filter_col: str, filter_value: str, description: str, cache_key_prefix: str):
    """Helper function to display filtered records with pagination and session-state caching.
    
    Args:
        df: DataFrame to filter
        filter_col: Column to filter on
        filter_value: Value to filter for
        description: Description for captions
        cache_key_prefix: Prefix for session state keys to avoid collisions
    """
    # Generate filtered dataset cache key
    filtered_df_key = f"{cache_key_prefix}_df"
    
    # Get or compute filtered DataFrame
    if filtered_df_key not in st.session_state:
        filtered_df = df[df[filter_col] == filter_value]
        st.session_state[filtered_df_key] = filtered_df
    else:
        filtered_df = st.session_state[filtered_df_key]
    
    if not filtered_df.empty:
        st.caption(f"Found {len(filtered_df)} records {description} (from sample)")
        
        # Add pagination
        page_size = 100
        total_rows = len(filtered_df)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        
        # Use session state for page persistence
        page_key = f"{cache_key_prefix}_page"
        if page_key not in st.session_state:
            st.session_state[page_key] = 1
            
        if total_pages > 1:
            page = st.slider(
                "Page", 1, total_pages, 
                st.session_state[page_key],
                key=f"page_slider_{cache_key_prefix}",
                on_change=lambda: setattr(st.session_state, page_key, st.session_state[f"page_slider_{cache_key_prefix}"])
            )
        else:
            page = 1
            
        # Update session state
        st.session_state[page_key] = page
        
        # Calculate slice indices
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        
        # Display data for current page with caching
        display_key = f"{cache_key_prefix}_display_{page}"
        if display_key not in st.session_state:
            display_df = filtered_df.iloc[start_idx:end_idx]
            st.session_state[display_key] = display_df
        else:
            display_df = st.session_state[display_key]
            
        st.dataframe(display_df, use_container_width=True)
        st.caption(f"Showing {description} records {start_idx+1}-{end_idx} of {total_rows}")
    else:
        st.info(f"No records {description} in sample")

def display_mapping_effectiveness(mapping_comparison: Dict):
    """
    Display mapping effectiveness analysis using the entire matched dataset.
    
    Args:
        mapping_comparison: Dictionary containing mapping comparison statistics
    """
    # For lazy loading or async tasks
    if isinstance(mapping_comparison, str) and mapping_comparison.startswith("task_"):
        # This is a task ID, check status
        from async_utils import TaskManager, TaskStatus
        
        # Create a task manager instance
        task_manager = TaskManager()
        
        # Get task status
        task_status = task_manager.get_task_status(mapping_comparison)
        status = task_status["status"]
        
        # If task isn't complete, show progress and return
        if status != TaskStatus.COMPLETED.value:
            progress = task_status["progress"] or 0.0
            st.progress(progress, text=f"Calculating mapping effectiveness... {int(progress * 100)}%")
            
            if status == TaskStatus.FAILED.value:
                st.error(f"Analysis failed: {task_status['error']}")
            return
            
        # If completed, get the result
        mapping_comparison = task_manager.get_task_result(mapping_comparison)
    
    # Handle empty data
    if not mapping_comparison:
        st.info("No mapping comparison data available. This could be because there are no matched records or no mappings defined.")
        return
        
    # Use a unique cache key based on the mapping comparison content
    cache_key = f"mapping_effectiveness_{hash(str(mapping_comparison))}"
    
    st.subheader("Mapping Effectiveness Analysis")
    st.write("This analysis shows how well the source data maps to the target data using all matched records.")
    
    # Add note about full dataset
    st.caption("Analysis performed using the complete matched dataset for maximum accuracy.")
    
    # Create a summary table - Only compute once and cache in session state
    summary_key = f"{cache_key}_summary"
    if summary_key not in st.session_state:
        summary_data = []
        for mapping_key, stats in mapping_comparison.items():
            summary_data.append({
                "Mapping": mapping_key,
                "Records": stats['total_records'],
                "Exact Matches": stats['exact_matches'],
                "Match %": f"{stats['exact_match_percentage']:.2f}%",
                "Avg Similarity": f"{stats.get('avg_similarity_non_matches', 0):.2f}%" if stats.get('avg_similarity_non_matches') is not None else "N/A"
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.session_state[summary_key] = summary_df
        else:
            st.session_state[summary_key] = None
    
    summary_df = st.session_state[summary_key]
    
    if summary_df is not None:
        # Style the summary table - Keep this part as the data is small
        def style_percentage(val):
            if isinstance(val, str) and val.endswith('%'):
                try:
                    percentage = float(val.rstrip('%'))
                    # Color gradient from red to yellow to green
                    if percentage >= 90:
                        return f'background-color: #c6efce; color: #006100'  # Green
                    elif percentage >= 70:
                        return f'background-color: #ffeb9c; color: #9c5700'  # Yellow
                    else:
                        return f'background-color: #ffc7ce; color: #9c0006'  # Red
                except ValueError:
                    pass
            return ''
        
        # Apply styling
        styled_df = summary_df.style.map(style_percentage, subset=['Match %', 'Avg Similarity'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Add a bar chart using cached data - FIX: Use a container with key instead of adding key to bar_chart
        chart_key = f"{cache_key}_chart"
        if chart_key not in st.session_state:
            chart_data = pd.DataFrame({
                'Mapping': summary_df['Mapping'],
                'Match %': summary_df['Match %'].str.rstrip('%').astype(float)
            }).sort_values('Match %', ascending=False)
            st.session_state[chart_key] = chart_data
        
        # Create a container with a key for uniqueness
        with st.container():
            st.bar_chart(st.session_state[chart_key].set_index('Mapping'), use_container_width=True)
        
        # Add detailed analysis section - use selectbox for lazy loading
        st.subheader("Detailed Mapping Analysis")
        mapping_options = list(mapping_comparison.keys())
        
        if mapping_options:
            # Use session state to remember selected mapping
            selected_key = f"{cache_key}_selected"
            if selected_key not in st.session_state:
                st.session_state[selected_key] = mapping_options[0]
                
            # Selectbox for mapping selection
            selected_mapping = st.selectbox(
                "Select mapping to analyze", 
                mapping_options,
                index=mapping_options.index(st.session_state[selected_key]),
                key=f"{cache_key}_selector",
                on_change=lambda: setattr(st.session_state, selected_key, st.session_state[f"{cache_key}_selector"])
            )
            
            # Update session state
            st.session_state[selected_key] = selected_mapping
            
            # Display details for selected mapping
            stats = mapping_comparison[selected_mapping]
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Records", stats['total_records'])
                st.metric("Exact Matches", stats['exact_matches'])
                
            with col2:
                st.metric("Match Rate", f"{stats['exact_match_percentage']:.2f}%")
                similarity_text = f"{stats.get('avg_similarity_non_matches', 0):.2f}%" if stats.get('avg_similarity_non_matches') is not None else "N/A"
                st.metric("Avg Similarity (non-matches)", similarity_text)
            
            # Sample comparison table
            samples_key = f"{cache_key}_samples_{selected_mapping}"
            if samples_key not in st.session_state:
                samples = stats.get('samples', [])
                if samples:
                    sample_rows = []
                    for i, sample in enumerate(samples):
                        sample_rows.append({
                            "#": i+1,
                            "Source Value": sample['source'],
                            "Target Value": sample['target'],
                            "Match": "✓" if sample['match'] else "✗"
                        })
                    st.session_state[samples_key] = pd.DataFrame(sample_rows) if sample_rows else None
                else:
                    st.session_state[samples_key] = None
            
            # Display samples if available
            sample_df = st.session_state[samples_key]
            if sample_df is not None:
                st.subheader("Sample Comparisons")
                
                # Custom styling for the match column
                def style_match(val):
                    return 'color: green; font-weight: bold' if val == '✓' else 'color: red; font-weight: bold'
                
                st.dataframe(
                    sample_df.style.map(style_match, subset=['Match']),
                    use_container_width=True
                )
    
    # Add detailed column-by-column analysis
    if mapping_comparison and isinstance(mapping_comparison, dict) and mapping_comparison:
        st.subheader("Detailed Column Mapping Analysis")
        
        # Create a tab for each mapped column
        column_tabs = st.tabs(list(mapping_comparison.keys()))
        
        # Display detail for each column
        for i, (col_name, tab) in enumerate(zip(mapping_comparison.keys(), column_tabs)):
            with tab:
                col_stats = mapping_comparison[col_name]
                
                # Overview statistics for this column
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Records", f"{col_stats['total_records']:,}")
                col2.metric("Exact Matches", f"{col_stats['exact_matches']:,}")
                col3.metric("Match Rate", f"{col_stats['exact_match_percentage']:.2f}%")
                
                similarity = col_stats.get('avg_similarity_non_matches')
                col4.metric("Avg Similarity", f"{similarity:.2f}%" if similarity is not None else "N/A")
                
                # Show top non-matching values
                if 'top_non_matches' in col_stats:
                    st.subheader("Top Non-Matching Values")
                    
                    top_values = col_stats['top_non_matches']
                    if top_values:
                        # Create dataframe for display
                        top_df = pd.DataFrame(top_values)
                        
                        # Add frequency column if available
                        if 'frequency' in top_df.columns:
                            # Sort by frequency descending
                            top_df = top_df.sort_values('frequency', ascending=False)
                        
                        st.dataframe(top_df, use_container_width=True)
                    else:
                        st.info("No non-matching values recorded")
                
                # Value distribution by category if available
                if 'value_distribution' in col_stats:
                    st.subheader("Value Distribution")
                    
                    dist_data = col_stats['value_distribution']
                    dist_df = pd.DataFrame(list(dist_data.items()), columns=["Value", "Count"])
                    dist_df = dist_df.sort_values("Count", ascending=False).head(10)
                    
                    fig = px.bar(
                        dist_df,
                        x="Value",
                        y="Count",
                        title="Top 10 Values",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True, key=f"dist_chart_{col_name}_{i}")

def display_async_task_status(task_id: str, auto_refresh: bool = True, 
                             key_prefix: str = "task_status", show_result: bool = False):
    """Display the status of an async task with progressive updates.
    
    Args:
        task_id: ID of the task to display
        auto_refresh: Whether to automatically refresh the status
        key_prefix: Prefix for session state keys
        show_result: Whether to show the task result when complete
    """
    import time
    from async_utils import TaskManager, TaskStatus
    
    if not task_id:
        return None
    
    # Create status container
    status_container = st.empty()
    
    # Create task manager instance
    task_manager = TaskManager()
    
    # Get task status
    task_status = task_manager.get_task_status(task_id)
    status = task_status["status"]
    progress = task_status["progress"] or 0.0
    
    # Update session state if needed
    refresh_key = f"{key_prefix}_{task_id}_refresh_count"
    if refresh_key not in st.session_state:
        st.session_state[refresh_key] = 0
    
    # Display progress based on status
    with status_container:
        if status == TaskStatus.PENDING.value:
            st.info("Task queued, waiting to start...")
        elif status == TaskStatus.RUNNING.value:
            st.progress(progress, text=f"Processing... {int(progress * 100)}%")
            if auto_refresh:
                st.session_state[refresh_key] += 1
                time.sleep(0.1)  # Brief pause
                st.rerun()
        elif status == TaskStatus.COMPLETED.value:
            st.success("Task completed successfully")
            if show_result:
                result = task_manager.get_task_result(task_id)
                return result
            return True
        elif status == TaskStatus.FAILED.value:
            st.error(f"Task failed: {task_status['error']}")
            return False
    
    return None

def display_async_results_when_ready(task_id: str, display_func: Callable, *args, **kwargs):
    """Display results when async task completes, with automatic refresh.
    
    Args:
        task_id: ID of the task to wait for
        display_func: Function to display results once complete
        args, kwargs: Arguments to pass to the display function
    """
    import time
    from async_utils import TaskManager
    
    # Create task manager instance
    task_manager = TaskManager()
    
    # Check if task is complete
    if task_manager.is_task_complete(task_id):
        # Get the result
        result = task_manager.get_task_result(task_id)
        
        # Display the result
        if result is not None:
            return display_func(result, *args, **kwargs)
        else:
            st.warning("Task completed but no results are available")
    else:
        # Show progress and refresh
        refresh_key = f"refresh_{task_id}"
        if refresh_key not in st.session_state:
            st.session_state[refresh_key] = 0
            
        # Get status
        task_status = task_manager.get_task_status(task_id)
        progress = task_status["progress"] or 0.0
        
        # Show progress
        st.progress(progress, text=f"Processing... {int(progress * 100)}%")
        
        # Auto-refresh
        st.session_state[refresh_key] += 1
        time.sleep(0.1)
        st.rerun()
        
    return None

def display_validation_summary(validation_results: list, pass_threshold: float = 95.0):
    """Display validation results summary.
    
    Args:
        validation_results: List of validation results
        pass_threshold: Threshold for pass/fail coloring (default: 95.0%)
    """
    st.markdown('<h3 class="step-header">Validation Results Summary</h3>', unsafe_allow_html=True)
    summary_df = pd.DataFrame(validation_results)
    
    # Fix for the deprecated Styler.applymap warning
    def style_validation_results(val):
        if isinstance(val, str) and val.endswith('%'):
            percentage = float(val.rstrip('%'))
            color = '#27AE60' if percentage >= pass_threshold else '#E74C3C'
            return f'color: {color}; font-weight: bold'
        return ''
    
    # Use .map instead of .applymap (deprecated)
    st.dataframe(
        summary_df.style.map(style_validation_results),
        use_container_width=True
    )

def display_detailed_validation_results(df: pd.DataFrame, validation_results: list, validation_rules: dict):
    """Display detailed validation results with failing records.
    
    Args:
        df: Target dataframe
        validation_results: List of validation result dictionaries
        validation_rules: Validation rules configuration
    """
    from utils import clean_dataframe_for_display
    
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
                
                if 'filtered_df' in locals() and filtered_df is not None:
                    st.write(f"Failed records count: {len(filtered_df)}")
                    display_df = clean_dataframe_for_display(filtered_df)
                    st.dataframe(display_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying validation results for column {col}: {str(e)}")
            
            st.write("---")

def display_business_rule_violations(df: pd.DataFrame, violations: Dict[str, List[int]], rules: List[Dict], format_rule_func: Callable):
    """Display business rule violations.
    
    Args:
        df: Target dataframe
        violations: Dictionary mapping rule names to lists of violating row indices
        rules: List of business rule configurations
        format_rule_func: Function to format rule as readable sentence
    """
    if not violations:
        st.success("All business rules passed!")
        return
        
    st.error("Business rules violations found:")
    for rule_name, rule_violations in violations.items():
        # Find the rule object
        rule = next((r for r in rules if r['name'] == rule_name), None)
        if rule:
            st.markdown(f"### Rule: {rule_name}")
            # Show rule in plain language
            st.info(format_rule_func(rule))
            
            # Show violations
            st.markdown(f"**Violations Found:** {len(rule_violations)}")
            if len(rule_violations) > 0:
                with st.expander("View Violations"):
                    st.dataframe(df.loc[rule_violations])

def display_report_summary(source_summary, target_summary, mapping, matching_count, validation_results, business_rules_info, report_timestamp, report_checksum):
    """
    Ultra-simple report summary that loads instantly with no processing.
    This function only displays already computed values with minimal formatting.
    """
    st.markdown("## Final Report Summary")
    
    # Basic metrics in columns for quick display
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Source Records", f"{source_summary.get('rows', 0):,}")
        st.metric("Target Records", f"{target_summary.get('rows', 0):,}")
        mapped_columns = len(mapping.get('mappings', {}))
        st.metric("Mapped Columns", mapped_columns)
    
    with col2:
        st.metric("Matched Records", f"{matching_count:,}")
        if validation_results:
            st.metric("Validation Rules", len(validation_results))
        if business_rules_info:
            st.metric("Business Rules", business_rules_info.get('count', 0))
    
    # Simple timestamp info
    st.markdown("---")
    st.markdown(f"**Report generated:** {report_timestamp}")
    
    # Add download buttons
    if st.button("Generate PDF Report"):
        st.info("To download complete report data, use the Export options in each section")

def soda_config_form(validation_config: Dict, business_rules: List, table_name_default: str = "your_table"):
    """Form for SODA CLI configuration generation.
    
    Args:
        validation_config: Validation rules configuration
        business_rules: Business rules configuration
        table_name_default: Default table name
        
    Returns:
        Table name entered by user
    """
    table_name = st.text_input("Table name for SODA checks", table_name_default)
    
    # Debug info to verify the rules being passed
    if st.checkbox("Debug validation rules"):
        st.write("Validation rules:", validation_config)
        st.write("Business rules:", business_rules)
        
    return table_name

def display_soda_yaml(yaml_content: str):
    """Display SODA YAML configuration with download button.
    
    Args:
        yaml_content: Generated YAML configuration
    """
    # Display the generated YAML
    st.code(yaml_content, language="yaml")
    
    # Add download button for SODA configuration
    st.download_button(
        label="Download SODA Configuration",
        data=yaml_content,
        file_name="soda_checks.yml",
        mime="text/yaml"
    )

def display_download_pdf_button(pdf_data: bytes, filename: str = "audit_report.pdf"):
    """Display download button for PDF report.
    
    Args:
        pdf_data: PDF file data as bytes
        filename: Download filename
    """
    st.download_button(
        "Download Audit Report (PDF)", 
        data=pdf_data, 
        file_name=filename, 
        mime="application/pdf"
    )

def display_download_csv_button(df: pd.DataFrame, filename: str):
    """Display download button for CSV data.
    
    Args:
        df: DataFrame to download
        filename: Download filename
    """
    from utils import clean_dataframe_for_display
    
    # Clean the DataFrame for pyarrow compatibility
    clean_df = clean_dataframe_for_display(df)
    csv = clean_df.to_csv(index=False)    
    
    st.download_button(
        "Download CSV",     
        data=csv, 
        file_name=filename, 
        mime="text/csv"
    )

def show_spinner_with_message(message: str):
    """Display spinner with custom message.
    
    Args:
        message: Message to display
        
    Returns:
        Spinner context manager
    """
    return st.spinner(message)

def string_transformation_form(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Form for selecting string transformations.
    
    Args:
        df: DataFrame to transform
    
    Returns:
        Dictionary mapping column names to list of transformations
    """
    st.subheader("String Transformations")
    st.markdown("""
    Apply transformations to text columns:
    - **Trim** - Remove leading and trailing whitespace
    - **Upper** - Convert to uppercase
    - **Lower** - Convert to lowercase
    """)
    transformations = {}
    
    # Get text columns
    text_columns = [col for col in df.columns if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])]
    
    if not text_columns:
        st.info("No text columns found in the dataset.")
        return transformations
    
    # Allow selection of columns to transform
    selected_columns = st.multiselect(
        "Select columns to transform",
        options=text_columns,
        default=[]
    )
    
    if not selected_columns:
        return transformations
    
    # Create a grid layout for transformation options
    cols = st.columns(3)
    with cols[0]:
        trim = st.checkbox("Trim whitespace", value=True, key="trim_transform")
    with cols[1]:
        upper = st.checkbox("Convert to UPPERCASE", key="upper_transform")
    with cols[2]:
        lower = st.checkbox("Convert to lowercase", key="lower_transform")
    
    # Build transformations dictionary based on selections
    for col in selected_columns:
        column_transforms = []
        if trim:
            column_transforms.append('trim')
        if upper and not lower:  # Don't apply both upper and lower
            column_transforms.append('upper')
        elif lower and not upper:
            column_transforms.append('lower')
        
        if column_transforms:
            transformations[col] = column_transforms
    
    # Show a preview of transformations that will be applied
    if transformations:
        st.subheader("Transformation Preview")
        preview_data = []
        for col, transforms in transformations.items():
            preview_data.append({
                "Column": col,
                "Transformations": ", ".join(transforms)
            })
        st.table(pd.DataFrame(preview_data))
    
    return transformations

# Add these new functions for optimized report display

@st.cache_data(ttl=600)
def display_validation_summary_charts(validation_results):
    """Display cached validation summary charts."""
    st.subheader("Data Quality Summary")
    
    # Create summary metrics for validation
    pass_count = sum(1 for result in validation_results if result.get("Pass", 0) > result.get("Fail", 0))
    fail_count = sum(1 for result in validation_results if result.get("Pass", 0) <= result.get("Fail", 0))
    total_validations = len(validation_results)
    
    # Donut chart for validation results
    if total_validations > 0:
        import plotly.graph_objects as go
        
        labels = ['Passed', 'Failed']
        values = [pass_count, fail_count]
        colors = ['#27AE60', '#E74C3C']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker=dict(colors=colors)
        )])
        
        fig.update_layout(
            title_text="Validation Rules Summary",
            annotations=[dict(text=f"{pass_count}/{total_validations}", x=0.5, y=0.5, font_size=20, showarrow=False)],
            height=300,
            margin=dict(t=40, b=0, l=0, r=0),
        )
        
        st.plotly_chart(fig, use_container_width=True)

@st.cache_data(ttl=600)
def display_business_rules_summary(business_rules, business_rule_violations):
    """Display cached business rules summary."""
    st.subheader("Business Rules Summary")
    
    # Count rules with violations
    rules_with_violations = len(business_rule_violations)
    rules_passing = len(business_rules) - rules_with_violations
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rules", len(business_rules))
    with col2:
        st.metric("Rules Passing", rules_passing, help="Rules without violations")
    
    # Display violations summary if any exist
    if rules_with_violations > 0:
        st.warning(f"{rules_with_violations} rules have violations")
        
        # Count total violations
        total_violations = sum(len(violations) for violations in business_rule_violations.values())
        st.metric("Total Violations", total_violations)

# Create cached version of the display function
@st.cache_data(ttl=600)
def _display_report_summary_cached(source_summary, target_summary, mapping, 
                                  matching_count, validation_results, business_rules_info,
                                  report_timestamp, report_checksum):
    """Cached implementation of report summary display."""
    # Summary Card
    with st.container():
        st.markdown("""
        <style>
        .summary-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .summary-header {
            color: #FF6600;
            margin-bottom: 15px;
            font-size: 1.2rem;
            font-weight: bold;
        }
        .summary-section {
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #dee2e6;
        }
        .summary-metrics {
            display: flex;
            justify-content: space-between;
            flex-wrap: wrap;
        }
        .summary-metric {
            text-align: center;
            padding: 10px;
            min-width: 120px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #2C3E50;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #7F8C8D;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="summary-card">', unsafe_allow_html=True)
        
        st.markdown('<div class="summary-header">Report Overview</div>', unsafe_allow_html=True)
        
        # Data metrics section
        st.markdown('<div class="summary-section">', unsafe_allow_html=True)
        st.markdown('<div class="summary-metrics">', unsafe_allow_html=True)
        
        # Source metric
        st.markdown(f'''
        <div class="summary-metric">
            <div class="metric-value">{source_summary.get('rows', 0):,}</div>
            <div class="metric-label">Source Records</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Target metric
        st.markdown(f'''
        <div class="summary-metric">
            <div class="metric-value">{target_summary.get('rows', 0):,}</div>
            <div class="metric-label">Target Records</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Matched metric
        source_rows = source_summary.get('rows', 0)
        match_pct = round((matching_count / source_rows * 100), 1) if source_rows > 0 else 0
        st.markdown(f'''
        <div class="summary-metric">
            <div class="metric-value">{matching_count:,}</div>
            <div class="metric-label">Matched Records ({match_pct}%)</div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Column metrics
        st.markdown(f'''
        <div class="summary-metric">
            <div class="metric-value">{source_summary.get('columns', 0)}</div>
            <div class="metric-label">Source Columns</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown(f'''
        <div class="summary-metric">
            <div class="metric-value">{target_summary.get('columns', 0)}</div>
            <div class="metric-label">Target Columns</div>
        </div>
        ''', unsafe_allow_html=True)
        
        mapped_count = len(mapping.get('mappings', {}))
        st.markdown(f'''
        <div class="summary-metric">
            <div class="metric-value">{mapped_count}</div>
            <div class="metric-label">Mapped Columns</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close summary-metrics
        st.markdown('</div>', unsafe_allow_html=True)  # Close summary-section
        
        # Validation metrics section
        if validation_results:
            st.markdown('<div class="summary-section">', unsafe_allow_html=True)
            st.markdown('<h3 style="font-size: 1.1rem; margin-bottom: 10px;">Data Quality Results</h3>', unsafe_allow_html=True)
            
            # Calculate validation metrics
            total_validations = len(validation_results)
            pass_rules = sum(1 for r in validation_results if r.get("Pass", 0) > r.get("Fail", 0))
            pass_pct = round((pass_rules / total_validations * 100), 1) if total_validations > 0 else 0
            
            st.markdown('<div class="summary-metrics">', unsafe_allow_html=True)
            
            # Total validations
            st.markdown(f'''
            <div class="summary-metric">
                <div class="metric-value">{total_validations}</div>
                <div class="metric-label">Total Validations</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Passed validations
            st.markdown(f'''
            <div class="summary-metric">
                <div class="metric-value">{pass_rules}</div>
                <div class="metric-label">Passed ({pass_pct}%)</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Failed validations
            st.markdown(f'''
            <div class="summary-metric">
                <div class="metric-value">{total_validations - pass_rules}</div>
                <div class="metric-label">Failed</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close summary-metrics
            st.markdown('</div>', unsafe_allow_html=True)  # Close summary-section
            
        # Business rules section
        if business_rules_info and business_rules_info.get('count', 0) > 0:
            st.markdown('<div class="summary-section">', unsafe_allow_html=True)
            st.markdown('<h3 style="font-size: 1.1rem; margin-bottom: 10px;">Business Rules</h3>', unsafe_allow_html=True)
            
            # Calculate business rule metrics
            total_rules = business_rules_info.get('count', 0)
            violations_by_rule = business_rules_info.get('violations', {})
            rules_with_violations = len(violations_by_rule)
            total_violations = sum(len(violations) for violations in violations_by_rule.values())
            
            st.markdown('<div class="summary-metrics">', unsafe_allow_html=True)
            
            # Total rules
            st.markdown(f'''
            <div class="summary-metric">
                <div class="metric-value">{total_rules}</div>
                <div class="metric-label">Total Rules</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Rules with violations
            st.markdown(f'''
            <div class="summary-metric">
                <div class="metric-value">{rules_with_violations}</div>
                <div class="metric-label">Rules with Violations</div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Total violations
            st.markdown(f'''
            <div class="summary-metric">
                <div class="metric-value">{total_violations}</div>
                <div class="metric-label">Total Violations</div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)  # Close summary-metrics
            st.markdown('</div>', unsafe_allow_html=True)  # Close summary-section
        
        # Report metadata with styled footer
        st.markdown(f"""
        <div style="margin-top: 20px; padding-top: 20px; border-top: 1px solid #dee2e6; font-size: 0.8rem; color: #7F8C8D;">
            <div><strong>Report Generated:</strong> {report_timestamp}</div>
            <div><strong>Report Checksum:</strong> <span style="font-family: monospace; word-break: break-all;">{report_checksum}</span></div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close summary-card

# Add this new function for displaying match quality insights
def display_match_quality_insights(matching_results: pd.DataFrame, key_columns: List[str]):
    """
    Display detailed insights about matching quality and key column distribution.
    
    Args:
        matching_results: DataFrame containing matching results
        key_columns: List of key columns used for matching
    """
    if matching_results is None or matching_results.empty:
        st.warning("No matching results available for quality insights.")
        return
    
    st.subheader("Match Quality Insights")
    
    # Get sample if needed (for large datasets)
    if len(matching_results) > 10000:
        sample = matching_results.sample(10000)
    else:
        sample = matching_results
    
    # Analyze key column completeness
    st.markdown("### Key Column Analysis")
    
    key_stats = []
    for col in key_columns:
        if col in matching_results.columns:
            # Calculate completeness and uniqueness
            null_count = matching_results[col].isna().sum()
            total = len(matching_results)
            completeness = 100 - (null_count / total * 100)
            
            # Calculate uniqueness (expensive operation - use sample)
            unique_count = sample[col].nunique()
            uniqueness = (unique_count / len(sample) * 100)
            
            key_stats.append({
                "Column": col,
                "Completeness": f"{completeness:.2f}%",
                "Uniqueness": f"{uniqueness:.2f}%",
                "Null Count": null_count,
                "Quality Score": (completeness + uniqueness) / 2  # Simple quality score
            })
    
    if key_stats:
        # Display key column quality table
        key_df = pd.DataFrame(key_stats)
        st.dataframe(key_df, use_container_width=True)
        
        # Display key quality visualization
        fig = px.bar(
            key_df,
            x="Column",
            y="Quality Score",
            color="Quality Score",
            labels={"Quality Score": "Overall Quality (%)"},
            title="Key Column Quality Scores",
            color_continuous_scale="RdYlGn"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Show distribution of match types
    if "_merge" in matching_results.columns:
        match_counts = matching_results["_merge"].value_counts()
        
        # Create detailed match breakdown
        st.markdown("### Match Type Distribution")
        
        fig = px.pie(
            names=match_counts.index,
            values=match_counts.values,
            color=match_counts.index,
            color_discrete_map={
                "both": "#27AE60",         # Green
                "left_only": "#E74C3C",    # Red
                "right_only": "#F39C12"    # Yellow/Orange
            },
            title="Match Type Distribution"
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

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
            except AttributeError as e:
                # If compute() isn't available, this might be a pandas DataFrame
                logger.error(f"Error sampling Dask DataFrame: {str(e)}")
                
                # Fallback: return the sample without computing
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
            return pd.DataFrame(columns=df.columns)
        else:
            return pd.DataFrame()

def display_comprehensive_report(report_data: Dict[str, Any]):
    """
    Display a comprehensive summary report that consolidates all key information.
    
    Args:
        report_data: Dictionary containing all report components
    """
    st.markdown("# Data Quality Report")
    
    # 1. METRICS SECTION
    st.markdown("## Key Metrics")
    
    # Create three columns for main metrics
    col1, col2, col3 = st.columns(3)
    
    # Matching metrics
    with col1:
        st.markdown("### Matching")
        metrics = report_data.get("matching_metrics", {})
        match_pct = metrics.get("match_percentage", 0)
        
        # Use delta to show percentage change
        st.metric(
            "Match Rate", 
            f"{match_pct:.1f}%", 
            delta=None
        )
        
        st.markdown(f"""
        **Total Records:** {metrics.get('total_records', 0):,}
        
        **Matched:** {metrics.get('total_match', 0):,}
        
        **Missing in Source:** {metrics.get('missing_source', 0):,}
        
        **Missing in Target:** {metrics.get('missing_target', 0):,}
        """)
    
    # Validation metrics
    with col2:
        st.markdown("### Validation Rules")
        metrics = report_data.get("validation_metrics", {})
        pass_pct = metrics.get("pass_percentage", 0)
        
        st.metric(
            "Pass Rate", 
            f"{pass_pct:.1f}%",
            delta=None
        )
        
        st.markdown(f"""
        **Total Rules:** {metrics.get('total_rules', 0):,}
        
        **Passed:** {metrics.get('pass_rules', 0):,}
        
        **Failed:** {metrics.get('fail_rules', 0):,}
        """)
        
    # Business rules metrics
    with col3:
        st.markdown("### Business Rules")
        metrics = report_data.get("business_rules_metrics", {})
        pass_pct = metrics.get("pass_percentage", 0)
        
        st.metric(
            "Pass Rate", 
            f"{pass_pct:.1f}%",
            delta=None
        )
        
        st.markdown(f"""
        **Total Rules:** {metrics.get('total_rules', 0):,}
        
        **Passed:** {metrics.get('rules_passing', 0):,}
        
        **Failed:** {metrics.get('rules_with_violations', 0):,}
        
        **Violations:** {metrics.get('total_violations', 0):,}
        """)
    
    # 2. UNMATCHED RECORDS SECTION
    st.markdown("## Unmatched Records")
    
    unmatched_exports = report_data.get("unmatched_exports", None)
    if unmatched_exports:
        col1, col2 = st.columns(2)
        
        # Missing in Source
        with col1:
            st.markdown("### Missing in Source")
            missing_in_source = unmatched_exports.get("missing_in_source")
            if isinstance(missing_in_source, pd.DataFrame) and not missing_in_source.empty:
                st.markdown(f"**Count:** {len(missing_in_source):,}")
                st.dataframe(missing_in_source.head(5), use_container_width=True)
                
                # Download button
                st.download_button(
                    "Download Missing in Source CSV",
                    missing_in_source.to_csv(index=False),
                    "missing_in_source.csv",
                    "text/csv",
                    key="download_missing_source"
                )
            else:
                st.info("No records missing in source.")
        
        # Missing in Target
        with col2:
            st.markdown("### Missing in Target")
            missing_in_target = unmatched_exports.get("missing_in_target")
            if isinstance(missing_in_target, pd.DataFrame) and not missing_in_target.empty:
                st.markdown(f"**Count:** {len(missing_in_target):,}")
                st.dataframe(missing_in_target.head(5), use_container_width=True)
                
                # Download button
                st.download_button(
                    "Download Missing in Target CSV",
                    missing_in_target.to_csv(index=False),
                    "missing_in_target.csv",
                    "text/csv",
                    key="download_missing_target"
                )
            else:
                st.info("No records missing in target.")
    else:
        st.info("No unmatched records available.")
    
    # 3. PREMISES SECTION - Summary of configurations
    st.markdown("## Configuration Summary")
    
    premises = report_data.get("premises", {})
    
    # Display in expandable section
    with st.expander("Mapping Configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Source to Target Mapping")
            st.markdown(f"**Mapped Columns:** {premises.get('mapped_columns_count', 0)}")
            
            # Show key columns
            key_columns = premises.get("key_columns", {})
            if key_columns:
                st.markdown("#### Key Columns")
                st.markdown(f"**Source:** {', '.join(key_columns.get('source', []))}")
                st.markdown(f"**Target:** {', '.join(key_columns.get('target', []))}")
        
        with col2:
            # Show a sample of column mappings
            st.markdown("### Sample Column Mappings")
            mappings = premises.get("mapping_config", {}).get("mappings", {})
            
            if mappings:
                # Convert a few mappings to a dataframe for display
                mapping_sample = []
                for i, (src_col, config) in enumerate(mappings.items()):
                    if i >= 5:  # Only show up to 5 examples
                        break
                    mapping_sample.append({
                        "Source Column": src_col,
                        "Target Column(s)": ", ".join(config.get("destinations", [])),
                        "Mapping Type": config.get("function", "Direct")
                    })
                
                if mapping_sample:
                    st.dataframe(pd.DataFrame(mapping_sample), hide_index=True)
                    
                    if len(mappings) > 5:
                        st.caption(f"Showing 5 of {len(mappings)} mappings")
    
    # 4. VALIDATION AND BUSINESS RULES SECTION
    st.markdown("## Data Quality Rules")
    
    # SODA Export
    st.markdown("### SODA Configuration Export")
    soda_config = report_data.get("soda_config", "")
    if soda_config:
        st.code(soda_config, language="yaml")
        
        # Download button for SODA config
        st.download_button(
            "Download SODA Configuration",
            data=soda_config,
            file_name="data_quality_checks.yml",
            mime="text/yaml",
            key="download_soda"
        )
    else:
        st.info("No SODA configuration available.")
    
    # 5. REPORT METADATA
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Report Generated:** {report_data.get('timestamp', 'N/A')}")
    
    with col2:
        st.markdown(f"**Report ID:** {report_data.get('checksum', 'N/A')}")
    
    # Generate PDF option
    st.markdown("### Export Full Report")
    if st.button("Generate PDF Report", type="primary"):
        st.info("PDF generation would be triggered here. This is a placeholder for the actual PDF generation functionality.")
