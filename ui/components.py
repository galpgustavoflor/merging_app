import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import plotly.express as px

from utils import display_sample, compute_basic_stats, generate_null_value_plot, generate_profile_report

class DataDisplay:
    @staticmethod
    def show_metadata(df: pd.DataFrame, title: str):
        st.subheader(title)
        
        # Display sample data
        st.write("Data Preview:")
        st.dataframe(display_sample(df))
        
        with st.spinner("Computing basic statistics..."):
            stats = compute_basic_stats(df)
            DataDisplay._show_basic_metrics(stats)
            DataDisplay._show_null_values(stats, title)
            DataDisplay._show_profile_report(df, title)

    @staticmethod
    def _show_basic_metrics(stats: Dict[str, Any]):
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

    @staticmethod
    def _show_null_values(stats: Dict[str, Any], title: str):
        with st.spinner("Generating null values visualization..."):
            fig = generate_null_value_plot(stats['null_counts'], title)
            st.plotly_chart(fig)

    @staticmethod
    def _show_profile_report(df: pd.DataFrame, title: str):
        show_profile_report = st.checkbox("Show Detailed Profile Report", key=f"show_profile_{title}")
        if show_profile_report:
            if 'profile_report' not in st.session_state:
                with st.spinner("Generating detailed profile report..."):
                    profile = generate_profile_report(df, title)
                    st.session_state.profile_report = profile.to_html()
            st.components.v1.html(st.session_state.profile_report, height=600, scrolling=True)

class ValidationDisplay:
    @staticmethod
    def show_validation_results(results: Dict[str, Any], df: Optional[pd.DataFrame] = None):
        st.subheader("Validation Results")
        
        # Show summary
        summary_df = pd.DataFrame(results)
        styled_df = ValidationDisplay._style_validation_results(summary_df)
        st.dataframe(styled_df)
        
        # Show details if DataFrame is provided
        if df is not None:
            ValidationDisplay._show_detailed_results(results, df)

    @staticmethod
    def _style_validation_results(df: pd.DataFrame):
        def color_pass_rate(val):
            if isinstance(val, str) and val.endswith('%'):
                percentage = float(val.rstrip('%'))
                return f"background-color: {'green' if percentage >= 95 else 'red'}"
            return ''
        
        return df.style.applymap(color_pass_rate, subset=['Pass Rate'])

    @staticmethod
    def _show_detailed_results(results: Dict[str, Any], df: pd.DataFrame):
        for result in results:
            if result.get("Fail", 0) > 0:
                with st.expander(f"Details for {result['Column']} - {result['Rule']}"):
                    st.write(f"Failed records: {result['Fail']}")
                    # Show failed records based on rule type
                    # Implementation depends on specific validation rules
