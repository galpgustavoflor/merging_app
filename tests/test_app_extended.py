import unittest
from unittest.mock import patch, MagicMock, Mock, call
import pandas as pd
import numpy as np
import streamlit as st
import sys
from pathlib import Path
import json
import plotly.express as px
from io import StringIO
import hashlib
from fpdf import FPDF

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app import *
from constants import Step, Column, ValidationRule as VRule

class TestAppExtended(unittest.TestCase):
    def setUp(self):
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Create test data with more variety
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3, None, 5],
            'name': ['A', 'B', 'B', None, 'E'],
            'value': [10.0, 20.0, -30.0, 40.0, 1000.0],
            'date': pd.date_range('2023-01-01', periods=5),
            'category': ['X', 'Y', 'Z', 'X', 'W']
        })
        
        # Initialize session state with test data
        st.session_state.df_source = self.test_df.copy()
        st.session_state.df_target = self.test_df.copy()
        st.session_state.step = Step.SOURCE_UPLOAD.value

    def test_handle_validation_rules_with_all_types(self):
        """Test validation rules handling with all types of validations"""
        st.session_state.matching_results = self.test_df
        st.session_state.df_target = self.test_df
        
        with patch('streamlit.checkbox', return_value=True), \
             patch('streamlit.text_input', return_value="A,B,C"), \
             patch('streamlit.number_input', side_effect=[0, 100]), \
             patch('streamlit.markdown'), \
             patch('streamlit.expander') as mock_expander:
            
            mock_expander.return_value.__enter__.return_value = MagicMock()
            handle_validation_rules()
            
            rules = st.session_state.validation_rules.get('id', {})
            self.assertTrue(all(rule in rules for rule in [
                VRule.VALIDATE_NULLS.value,
                VRule.VALIDATE_UNIQUENESS.value,
                VRule.VALIDATE_LIST_OF_VALUES.value,
                VRule.VALIDATE_RANGE.value
            ]))

    def test_display_metadata_with_errors(self):
        """Test metadata display with error handling"""
        with patch('streamlit.markdown'), \
             patch('streamlit.error') as mock_error, \
             patch('streamlit.tabs', side_effect=Exception("Test error")):
            
            display_metadata(self.test_df, "Test Data")
            mock_error.assert_called_once()

    @patch('streamlit.markdown')
    @patch('streamlit.write')
    def test_handle_matching_execution_with_errors(self, mock_write, mock_markdown):
        """Test matching execution with error handling"""
        st.session_state.key_source = ['nonexistent']
        st.session_state.key_target = ['nonexistent']
        
        with patch('utils.execute_matching_dask', side_effect=Exception("Test error")), \
             patch('streamlit.error') as mock_error:
            handle_matching_execution()
            mock_error.assert_called_once()

    def test_display_validation_results_with_all_scenarios(self):
        """Test validation results display with all possible scenarios"""
        validation_results = [
            {
                Column.NAME.value: 'id',
                'Rule': 'Null values',
                'Pass': 4,
                'Fail': 1,
                'Pass %': '80.00%'
            },
            {
                Column.NAME.value: 'name',
                'Rule': 'Unique values',
                'Pass': 3,
                'Fail': 2,
                'Pass %': '60.00%'
            }
        ]
        
        validation_rules = {
            'id': {
                VRule.VALIDATE_NULLS.value: True,
                VRule.VALIDATE_RANGE.value: True,
                VRule.MIN_VALUE.value: 0,
                VRule.MAX_VALUE.value: 100
            },
            'name': {
                VRule.VALIDATE_UNIQUENESS.value: True,
                VRule.VALIDATE_LIST_OF_VALUES.value: ['A', 'B', 'C']
            }
        }
        
        with patch('streamlit.markdown'), \
             patch('streamlit.dataframe'), \
             patch('streamlit.write'), \
             patch('streamlit.subheader'):
            
            display_validation_summary(validation_results)
            display_detailed_validation_results(self.test_df, validation_results, validation_rules)

    def test_generate_pdf_report_with_all_sections(self):
        """Test PDF report generation with all possible sections"""
        report_content = json.dumps({
            'summary': 'test',
            'details': {'key': 'value'}
        })
        
        source_summary = {
            'rows': 5,
            'columns': 5,
            'checksum': 'abc123'
        }
        
        target_summary = {
            'rows': 5,
            'columns': 5,
            'checksum': 'def456'
        }
        
        mapping = {
            'key_source': ['id'],
            'key_target': ['id'],
            'mappings': {
                'name': {
                    'destinations': ['name'],
                    'function': 'direct_mapping'
                },
                'value': {
                    'destinations': ['value'],
                    'function': 'conversion_mapping',
                    'transformation': '{"A": 1}'
                }
            }
        }
        
        validation_rules = {
            'id': {
                VRule.VALIDATE_NULLS.value: True,
                VRule.VALIDATE_RANGE.value: True
            },
            'name': {
                VRule.VALIDATE_UNIQUENESS.value: True
            }
        }
        
        validation_results = [
            {
                Column.NAME.value: col,
                'Rule': rule,
                'Pass': 4,
                'Fail': 1,
                'Pass %': '80.00%'
            }
            for col in ['id', 'name']
            for rule in ['Null values', 'Unique values']
        ]
        
        pdf_data = generate_pdf_report(
            report_content,
            source_summary,
            target_summary,
            "Test checksum note",
            mapping,
            self.test_df,
            validation_rules,
            validation_results
        )
        
        self.assertIsInstance(pdf_data, bytes)
        self.assertTrue(len(pdf_data) > 0)

    def test_handle_report_summary_with_all_data(self):
        """Test report summary handling with all possible data"""
        st.session_state.df_source = self.test_df
        st.session_state.df_target = self.test_df
        st.session_state.mapping = {
            'key_source': ['id'],
            'key_target': ['id'],
            'mappings': {'name': {'destinations': ['name']}}
        }
        st.session_state.matching_results = self.test_df
        st.session_state.validation_results = [
            {
                Column.NAME.value: 'id',
                'Rule': rule,
                'Pass': 4,
                'Fail': 1,
                'Pass %': '80.00%'
            }
            for rule in ['Null values', 'Unique values']
        ]
        
        with patch('streamlit.markdown'), \
             patch('streamlit.write'), \
             patch('streamlit.download_button'), \
             patch('streamlit.subheader'), \
             patch('streamlit.dataframe'), \
             patch('streamlit.info'), \
             patch('streamlit.json'), \
             patch('app.generate_pdf_report') as mock_pdf:
            
            mock_pdf.return_value = b"test pdf content"
            handle_report_summary()
            mock_pdf.assert_called_once()

    def test_step_navigation_state_changes(self):
        """Test step navigation with state changes"""
        for step in Step:
            st.session_state.step = step.value
            
            with patch('streamlit.columns') as mock_cols, \
                 patch('streamlit.button') as mock_button:
                
                mock_cols.return_value = [MagicMock(), MagicMock()]
                mock_button.return_value = True
                
                step_navigation()
                
                if step.value < len(Step):
                    self.assertEqual(st.session_state.step, step.value + 1)

    def test_render_breadcrumbs_with_interactions(self):
        """Test breadcrumb rendering with user interactions"""
        for step in Step:
            st.session_state.step = step.value
            
            with patch('streamlit.container'), \
                 patch('streamlit.markdown'), \
                 patch('streamlit.columns') as mock_cols, \
                 patch('streamlit.button') as mock_button:
                
                mock_cols.return_value = [MagicMock() for _ in range(len(Step))]
                mock_button.return_value = True
                
                render_breadcrumbs()
                
                if step.value > 1:
                    mock_button.assert_called()

if __name__ == '__main__':
    unittest.main()
