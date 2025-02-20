import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Dict, Any, Optional
from config.settings import APP_CONFIG, DATAHUB_CONFIG
from core.file_processor import FileProcessor
from core.data_validator import DataValidator
from utils.logging_utils import setup_logging, get_logger
from core.datahub_processor import DataHubProcessor

# Initialize logger
logger = get_logger(__name__)

def initialize_datahub_connection(username: str, password: str) -> Optional[DataHubProcessor]:
    try:
        processor = DataHubProcessor(username, password)
        if processor.connect():
            return processor
        return None
    except Exception as e:
        logger.error(f"DataHub error: {str(e)}")
        return None

def display_summary_tab(summary: Dict[str, Any]) -> None:
    """Display enhanced summary information."""
    try:
        # Dataset Overview
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{summary['shape'][0]:,}")
        with col2:
            st.metric("Sample Size", f"{summary['sample_shape'][0]:,}")
        with col3:
            st.metric("Total Columns", f"{summary['shape'][1]:,}")
        with col4:
            st.metric("Memory Usage", f"{summary['memory_usage']:.2f} MB")

        # Column Information with string conversion
        st.subheader("📋 Column Information")
        col_info = pd.DataFrame({
            'Column': summary['columns'],
            'Type': [summary['dtypes'][col] for col in summary['columns']],
            'Non-Null Count': [summary['sample'][col].count() for col in summary['columns']],
            'Memory Usage (KB)': [summary['sample'][col].memory_usage(deep=True)/1024 for col in summary['columns']]
        }).astype(str)  # Convert all to strings for display
        st.dataframe(col_info, hide_index=True)

        # Numeric Columns Distribution
        if summary.get('numeric_summary'):
            st.subheader("📈 Numeric Columns Distribution")
            numeric_cols = summary['sample'].select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                selected_col = st.selectbox("Select column for distribution:", numeric_cols)
                fig = px.histogram(summary['sample'], x=selected_col, title=f'Distribution of {selected_col}')
                st.plotly_chart(fig, use_container_width=True)

        # Sample Data Preview with updated styling
        st.subheader("🔍 Data Preview")
        styled_df = summary['sample'].style.highlight_null(
            props='background-color: #FFB6C1; color: #000000;'
        )
        st.dataframe(styled_df, height=300)
    except Exception as e:
        st.error(f"Error displaying summary: {str(e)}")
        logger.error(f"Summary display error: {str(e)}")

def display_validation_tab(report: Dict[str, Any]) -> None:
    """Display validation results in a more structured way."""
    # Null Analysis
    st.subheader("📊 Null Value Analysis")
    null_df = pd.DataFrame([
        {
            'Column': col,
            'Null Count': data['count'],
            'Null Percentage': f"{data['percentage']:.2f}%",
            'Status': data['status']
        }
        for col, data in report['null_checks'].items()
    ])
    
    # Color code the status
    styled_null_df = null_df.style.apply(lambda x: ['background-color: #90EE90' if v == 'OK' 
                                                   else 'background-color: #FFB6C1' 
                                                   for v in x], subset=['Status'])
    st.dataframe(styled_null_df, hide_index=True)

    # Data Type Analysis
    st.subheader("📋 Data Type Analysis")
    dtype_df = pd.DataFrame([
        {
            'Column': col,
            'Data Type': data['dtype'],
            'Is Numeric': '✓' if data['numeric'] else '✗',
            'Is Temporal': '✓' if data['temporal'] else '✗'
        }
        for col, data in report['dtype_checks'].items()
    ])
    st.dataframe(dtype_df, hide_index=True)

def main():
    setup_logging()
    st.set_page_config(page_title=APP_CONFIG["name"], page_icon="📊", layout="wide")
    
    st.title("File Mapping Process")
    st.sidebar.header("Data Source")

    # Initialize session state
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'query_submitted' not in st.session_state:
        st.session_state.query_submitted = False
    if 'query_result' not in st.session_state:
        st.session_state.query_result = None

    # Data source selection
    data_source = st.sidebar.radio("Select Data Source", ["File Upload", "DataHub Connection"])
    
    if data_source == "File Upload":
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
        if uploaded_file is not None:
            try:
                st.session_state.processor = FileProcessor(uploaded_file)
                st.session_state.data_loaded = st.session_state.processor.data is not None
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                logger.error(f"File processing error: {str(e)}")
    if data_source == "DataHub Connection":  # DataHub Connection
        st.sidebar.subheader("DataHub Authentication")
        username = st.sidebar.text_input("Username", type="default")
        password = st.sidebar.text_input("Password", type="password")
        
        
        if 'datahub_processor' not in st.session_state and st.button("Connect to DataHub"):
            if not username or not password:
                st.error("Please provide both username and password")
            else:
                with st.spinner('Connecting to DataHub...'):
                    try:
                        processor = initialize_datahub_connection(username, password)
                        if processor:
                            st.session_state.processor = processor
                            st.session_state['datahub_processor'] = processor
                            st.success("✅ Successfully connected to DataHub!")
                        else:
                            st.error("❌ Failed to connect to DataHub")
                            st.info("Please check your credentials and try again.")
                    except Exception as e:
                        st.error(f"DataHub connection error: {str(e)}")
                        logger.error(f"DataHub error: {str(e)}")

        # Show query interface if connected
        if 'datahub_processor' in st.session_state:
            processor = st.session_state.datahub_processor
            
            # Tables list in sidebar
            #try:
            #    with st.sidebar.expander("Available Tables"):
            #        tables_df = processor.get_tables()
            #        if tables_df is not None:
            #            st.dataframe(tables_df, height=200)
            #except Exception as e:
            #    st.sidebar.error("Could not fetch tables")
            
            # Query input and execution
            query = st.text_area("Enter SQL Query:")
            execute_button = st.button("Execute Query")
            
            if execute_button and query:
                with st.spinner('Executing query...'):
                    try:
                        st.session_state.query_result = processor.execute_query(query)
                        st.session_state.query_submitted = True
                    except Exception as e:
                        st.error(f"Query execution failed: {str(e)}")
            
            # Display query results
            if st.session_state.query_submitted and st.session_state.query_result is not None:
                try:
                    result_df = st.session_state.query_result.compute()  # Convert Dask DataFrame to Pandas
                    st.success("Query executed successfully!")
                    st.write("Query Results:")
                    st.dataframe(result_df)
                    st.session_state.data_loaded = result_df is not None
                    st.write(f"Total rows: {st.session_state.processor}")
                    # Add download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name="query_results.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Error displaying results: {str(e)}")
    
    # Process data if loaded successfully
    if st.session_state.data_loaded and st.session_state.processor is not None:
        
        try:
            with st.spinner('Processing data...'):
                # Show data source info
                st.subheader("Data Source Information")
                if data_source == "File Upload":
                    st.json({
                        "Source": "File Upload",
                        "filename": uploaded_file.name,
                        "size": f"{uploaded_file.size / 1024 / 1024:.2f} MB",
                        "type": uploaded_file.type
                    })
                else:
                    st.json({
                        "Source": "DataHub",
                        "Connection": f"{DATAHUB_CONFIG['host']}:{DATAHUB_CONFIG['port']}"
                    })
                
                # Create tabs and process data
                tab1, tab2, tab3 = st.tabs(["Summary", "Validation", "Profile"])
                
                with tab1:
                    st.subheader("Data Summary")
                    summary = st.session_state.processor.get_summary()
                    display_summary_tab(summary)

                with tab2:
                    st.subheader("Validation Results")
                    validator = DataValidator(st.session_state.processor.data)
                    if validator.validate():
                        st.success("✅ File validation passed!")
                    else:
                        st.error("❌ File validation failed!")
                    
                    report = validator.get_validation_report()
                    display_validation_tab(report)

                with tab3:
                    st.subheader("Data Profile")
                    if st.button("Generate Profile Report"):
                        with st.spinner('Generating profile report...'):
                            try:
                                profile_html = st.session_state.processor.generate_profile()
                                st.components.v1.html(profile_html, height=800, scrolling=True)
                            except Exception as e:
                                st.error(f"Error generating profile: {str(e)}")
                                logger.error(f"Profile generation error: {str(e)}")
                            
        except Exception as e:
            error_msg = f"Error processing data: {str(e)}"
            st.error(error_msg)
            logger.error(error_msg)

if __name__ == "__main__":
    main()
