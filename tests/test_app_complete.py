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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app import *
from constants import Step, Column, ValidationRule as VRule

class TestAppComplete(unittest.TestCase):
    def setUp(self):
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Create comprehensive test data
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3, None, 5],
            'name': ['A', 'B', 'B', None, 'E'],
            'value': [10.0, 20.0, -30.0, 40.0, 1000.0],
            'date': pd.date_range('2023-01-01', periods=5),
            'category': ['X', 'Y', 'Z', 'X', 'W']
        })

    @patch('streamlit.set_page_config')
    @patch('streamlit.container')
    @patch('app.inject_custom_css')
    @patch('app.render_breadcrumbs')
    @patch('streamlit.markdown')
    def test_main_initialization_complete(self, mock_markdown, mock_breadcrumbs, 
                                       mock_css, mock_container, mock_config):
        # Test main initialization with different states
        states = [Step.SOURCE_UPLOAD, Step.TARGET_UPLOAD, Step.MAPPING_RULES]
        for step in states:
            st.session_state.step = step.value
            main()
            mock_config.assert_called()
            mock_container.assert_called()

    def test_handle_source_file_upload_complete(self):
        with patch('streamlit.markdown'), \
             patch('streamlit.file_uploader') as mock_uploader, \
             patch('utils.FileLoader.load_file') as mock_load, \
             patch('app.display_metadata'), \
             patch('app.step_navigation'):
            
            # Test successful upload
            mock_file = MagicMock()
            mock_file.name = "test.csv"
            mock_uploader.return_value = mock_file
            mock_load.return_value = self.test_df
            
            handle_source_file_upload()
            self.assertIsNotNone(st.session_state.get('df_source'))
            
            # Test failed upload
            mock_load.return_value = None
            handle_source_file_upload()
            self.assertIsNone(st.session_state.get('df_source'))

    def test_handle_mapping_rules_complete(self):
        st.session_state.df_source = self.test_df
        st.session_state.df_target = self.test_df
        
        with patch('streamlit.markdown'), \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.file_uploader') as mock_uploader, \
             patch('streamlit.multiselect') as mock_multiselect, \
             patch('streamlit.radio') as mock_radio, \
             patch('streamlit.selectbox') as mock_select, \
             patch('streamlit.text_area') as mock_text, \
             patch('streamlit.checkbox') as mock_checkbox:
            
            # Test JSON upload
            mock_json = MagicMock()
            mock_json.read.return_value = '{"key_source": ["id"]}'
            mock_uploader.return_value = mock_json
            
            # Test column mapping
            mock_multiselect.side_effect = [['id'], ['id']]
            mock_radio.return_value = "Map"
            mock_select.return_value = Functions.DIRECT.value
            mock_expander.return_value.__enter__.return_value = MagicMock()
            mock_checkbox.return_value = True
            
            handle_mapping_rules()
            
            self.assertIn('key_source', st.session_state)
            self.assertIn('key_target', st.session_state)
            self.assertIn('mapping', st.session_state)

    def test_handle_validation_rules_complete(self):
        st.session_state.matching_results = self.test_df
        st.session_state.df_target = self.test_df
        
        with patch('streamlit.markdown'), \
             patch('streamlit.expander') as mock_expander, \
             patch('streamlit.checkbox') as mock_checkbox, \
             patch('streamlit.text_input') as mock_text, \
             patch('streamlit.number_input') as mock_number:
            
            # Test all validation types
            mock_checkbox.return_value = True
            mock_text.return_value = "A,B,C"
            mock_number.side_effect = [0, 100]
            mock_expander.return_value.__enter__.return_value = MagicMock()
            
            handle_validation_rules()
            
            validation_rules = st.session_state.get('validation_rules', {})
            self.assertTrue(all(
                validation_rules['id'].get(rule.value, False)
                for rule in [VRule.VALIDATE_NULLS, VRule.VALIDATE_UNIQUENESS]
            ))

    def test_handle_data_validation_complete(self):
        st.session_state.df_target = self.test_df
        st.session_state.validation_rules = {
            'id': {
                VRule.VALIDATE_NULLS.value: True,
                VRule.VALIDATE_UNIQUENESS.value: True,
                VRule.VALIDATE_RANGE.value: True,
                VRule.MIN_VALUE.value: 0,
                VRule.MAX_VALUE.value: 100
            }
        }
        
        with patch('streamlit.markdown'), \
             patch('streamlit.error'), \
             patch('utils.DataValidator.execute_validation') as mock_validate, \
             patch('app.display_validation_summary'), \
             patch('app.display_detailed_validation_results'), \
             patch('app.step_navigation'):
            
            mock_validate.return_value = [{
                Column.NAME.value: 'id',
                'Rule': 'Null values',
                'Pass': 4,
                'Fail': 1,
                'Pass %': '80.00%'
            }]
            
            handle_data_validation()
            self.assertIsNotNone(st.session_state.get('validation_results'))

    def test_display_validation_results_complete(self):
        validation_results = [{
            Column.NAME.value: 'id',
            'Rule': rule,
            'Pass': 4,
            'Fail': 1,
            'Pass %': '80.00%'
        } for rule in [
            'Null values',
            'Unique values',
            'Values outside allowed list',
            'Values not matching regex',
            'Values out of range'
        ]]
        
        validation_rules = {
            'id': {
                VRule.VALIDATE_NULLS.value: True,
                VRule.VALIDATE_UNIQUENESS.value: True,
                VRule.VALIDATE_LIST_OF_VALUES.value: ['A', 'B'],
                VRule.VALIDATE_REGEX.value: r'\d+',
                VRule.VALIDATE_RANGE.value: True,
                VRule.MIN_VALUE.value: 0,
                VRule.MAX_VALUE.value: 100
            }
        }
        
        with patch('streamlit.markdown'), \
             patch('streamlit.dataframe'), \
             patch('streamlit.write'), \
             patch('streamlit.subheader'), \
             patch('streamlit.error'):
            
            display_validation_summary(validation_results)
            display_detailed_validation_results(
                self.test_df,
                validation_results,
                validation_rules
            )

    def test_handle_report_summary_complete(self):
        st.session_state.df_source = self.test_df
        st.session_state.df_target = self.test_df
        st.session_state.mapping = {
            'key_source': ['id'],
            'key_target': ['id'],
            'mappings': {
                'name': {
                    'destinations': ['name'],
                    'function': Functions.DIRECT.value
                }
            }
        }
        st.session_state.matching_results = self.test_df
        st.session_state.validation_results = [{
            Column.NAME.value: 'id',
            'Rule': 'Null values',
            'Pass': 4,
            'Fail': 1,
            'Pass %': '80.00%'
        }]
        
        with patch('streamlit.markdown'), \
             patch('streamlit.write'), \
             patch('streamlit.download_button'), \
             patch('streamlit.subheader'), \
             patch('streamlit.dataframe'), \
             patch('streamlit.json'), \
             patch('streamlit.info'), \
             patch('app.generate_pdf_report') as mock_pdf:
            
            mock_pdf.return_value = b"test pdf content"
            handle_report_summary()
            mock_pdf.assert_called_once()

    def test_error_handling_complete(self):
        """Test error handling in all major functions"""
        with patch('streamlit.error') as mock_error:
            # Test file loading error
            handle_source_file_upload()
            mock_error.assert_not_called()
            
            # Test mapping error
            handle_mapping_rules()
            mock_error.assert_called()
            
            # Test validation error
            handle_data_validation()
            mock_error.assert_called()

if __name__ == '__main__':
    unittest.main()
