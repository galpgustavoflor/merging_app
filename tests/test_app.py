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

class TestApp(unittest.TestCase):
    def setUp(self):
        # Reset session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Create test data
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C'],
            'value': [10.0, 20.0, 30.0],
            'date': pd.date_range('2023-01-01', periods=3)
        })
        
        # Initialize session state
        st.session_state.df_source = self.test_df.copy()
        st.session_state.df_target = self.test_df.copy()
        st.session_state.step = Step.SOURCE_UPLOAD.value
        st.session_state.mapping = {}
        st.session_state.validation_rules = {}

    @patch('streamlit.set_page_config')
    def test_main_initialization(self, mock_config):
        with patch('app.render_breadcrumbs'), \
             patch('app.inject_custom_css'), \
             patch('streamlit.container'), \
             patch('streamlit.markdown'):
            main()
            self.assertIn('step', st.session_state)
            self.assertIn('df_source', st.session_state)
            self.assertIn('df_target', st.session_state)
            self.assertIn('mapping', st.session_state)
            self.assertIn('validation_rules', st.session_state)
            mock_config.assert_called_once()

    @patch('streamlit.markdown')
    @patch('streamlit.file_uploader')
    def test_handle_source_file_upload_success(self, mock_uploader, mock_markdown):
        mock_file = MagicMock()
        mock_file.name = "test.csv"
        mock_uploader.return_value = mock_file
        
        with patch('utils.FileLoader.load_file', return_value=self.test_df), \
             patch('app.display_metadata'), \
             patch('app.step_navigation'):
            handle_source_file_upload()
            self.assertIsNotNone(st.session_state.get('df_source'))

    @patch('streamlit.error')
    def test_handle_source_file_upload_error(self, mock_error):
        with patch('streamlit.file_uploader', return_value=MagicMock()) as mock_uploader, \
             patch('utils.FileLoader.load_file', side_effect=Exception("Test error")):
            handle_source_file_upload()
            mock_error.assert_called()

    def test_display_metadata_all_tabs(self):
        mock_tabs = [MagicMock() for _ in range(4)]
        with patch('streamlit.tabs', return_value=mock_tabs) as mock_tabs_func, \
             patch('streamlit.markdown'), \
             patch('streamlit.dataframe'), \
             patch('streamlit.write'), \
             patch('plotly.express.bar') as mock_px, \
             patch('streamlit.plotly_chart'):
            
            display_metadata(self.test_df, "Test Data")
            mock_tabs_func.assert_called_once()
            self.assertEqual(len(mock_tabs), 4)

    @patch('streamlit.markdown')
    @patch('streamlit.multiselect')
    @patch('streamlit.expander')
    @patch('streamlit.checkbox')
    @patch('streamlit.radio')
    @patch('streamlit.selectbox')
    def test_handle_mapping_rules_complete(self, mock_select, mock_radio, mock_check, 
                                         mock_expander, mock_multiselect, mock_markdown):
        mock_multiselect.side_effect = [['id'], ['id']]  # source and target keys
        mock_radio.return_value = "Map"
        mock_select.return_value = "direct_mapping"
        mock_expander.return_value.__enter__.return_value = MagicMock()
        
        handle_mapping_rules()
        self.assertIn('key_source', st.session_state)
        self.assertIn('key_target', st.session_state)
        self.assertEqual(st.session_state.key_source, ['id'])

    @patch('streamlit.markdown')
    @patch('streamlit.write')
    @patch('streamlit.dataframe')
    def test_handle_matching_execution_complete(self, mock_df, mock_write, mock_markdown):
        st.session_state.key_source = ['id']
        st.session_state.key_target = ['id']
        st.session_state.mapping = {'key_source': ['id'], 'key_target': ['id']}
        
        mock_merged = pd.DataFrame({'_merge': ['both', 'left_only', 'right_only']})
        with patch('utils.execute_matching_dask') as mock_match:
            mock_match.return_value = (mock_merged, {
                'total_match': 1,
                'missing_source': 1,
                'missing_target': 1
            })
            handle_matching_execution()
            mock_match.assert_called_once()
            self.assertIsNotNone(st.session_state.get('matching_results'))

    def test_handle_validation_rules_all_types(self):
        st.session_state.matching_results = self.test_df
        st.session_state.df_target = self.test_df
        
        mock_checks = {
            'nulls': True,
            'unique': True,
            'domain': True,
            'regex': True,
            'range': True
        }
        
        with patch('streamlit.checkbox', side_effect=lambda *args, **kwargs: mock_checks.get(kwargs.get('key', '').split('_')[0], False)), \
             patch('streamlit.text_input', return_value="test,values"), \
             patch('streamlit.number_input', return_value=0):
            handle_validation_rules()
            self.assertIsNotNone(st.session_state.get('validation_rules'))
            rules = st.session_state.validation_rules.get('id', {})
            self.assertTrue(rules.get(VRule.VALIDATE_NULLS.value))
            self.assertTrue(rules.get(VRule.VALIDATE_UNIQUENESS.value))

    def test_display_validation_results_complete(self):
        validation_results = [
            {
                Column.NAME.value: 'id',
                'Rule': rule,
                'Pass': 3,
                'Fail': 1,
                'Pass %': '75.00%'
            }
            for rule in ['Null values', 'Unique values', 'Values outside allowed list', 
                        'Values not matching regex', 'Values out of range']
        ]
        
        with patch('streamlit.markdown'), \
             patch('streamlit.dataframe'), \
             patch('streamlit.write'), \
             patch('streamlit.subheader'):
            display_validation_summary(validation_results)
            display_detailed_validation_results(
                self.test_df,
                validation_results,
                {
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
            )

    def test_generate_pdf_report_complete(self):
        report_content = json.dumps({'test': 'data'})
        source_summary = {'rows': 3, 'columns': 3, 'checksum': 'abc123'}
        target_summary = {'rows': 3, 'columns': 3, 'checksum': 'def456'}
        validation_results = [{
            Column.NAME.value: 'id',
            'Rule': 'Null values',
            'Pass': 3,
            'Fail': 0,
            'Pass %': '100.00%'
        }]
        
        pdf_data = generate_pdf_report(
            report_content,
            source_summary,
            target_summary,
            "Test checksum note",
            {'key_source': ['id'], 'mappings': {'col1': {'destinations': ['col2']}}},
            self.test_df,
            {'id': {VRule.VALIDATE_NULLS.value: True}},
            validation_results
        )
        self.assertIsInstance(pdf_data, bytes)
        self.assertTrue(len(pdf_data) > 0)

    def test_step_navigation_complete(self):
        for step in Step:
            st.session_state.step = step.value
            with patch('streamlit.columns', return_value=[MagicMock(), MagicMock()]) as mock_cols:
                step_navigation()
                mock_cols.assert_called_once()

    @patch('streamlit.markdown')
    @patch('streamlit.columns')
    def test_render_breadcrumbs_all_states(self, mock_columns, mock_markdown):
        mock_cols = [MagicMock() for _ in range(len(Step))]
        mock_columns.return_value = mock_cols
        
        for step in Step:
            st.session_state.step = step.value
            render_breadcrumbs()
            mock_columns.assert_called()

    def test_handle_report_summary_complete(self):
        st.session_state.df_source = self.test_df
        st.session_state.df_target = self.test_df
        st.session_state.mapping = {'key_source': ['id']}
        st.session_state.matching_results = self.test_df
        st.session_state.validation_results = [{
            Column.NAME.value: 'id',
            'Rule': 'Null values',
            'Pass': 3,
            'Fail': 0,
            'Pass %': '100.00%'
        }]
        
        with patch('streamlit.markdown'), \
             patch('streamlit.write'), \
             patch('streamlit.download_button'), \
             patch('streamlit.subheader'), \
             patch('streamlit.dataframe'), \
             patch('app.generate_pdf_report') as mock_pdf:
            mock_pdf.return_value = b"test pdf content"
            handle_report_summary()
            mock_pdf.assert_called_once()

if __name__ == '__main__':
    unittest.main()
