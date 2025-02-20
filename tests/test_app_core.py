import pytest
from unittest.mock import patch, MagicMock, Mock
import pandas as pd
import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app import handle_mapping_rules, handle_matching_execution, handle_validation_rules
from constants import Step, Functions

def test_handle_mapping_rules(mock_streamlit, sample_dataframe):
    """Test mapping rules handling with all components"""
    st.session_state.df_source = sample_dataframe
    st.session_state.df_target = sample_dataframe
    
    with patch('streamlit.expander') as mock_expander:
        mock_expander.return_value.__enter__.return_value = MagicMock()
        handle_mapping_rules()
        
        assert 'key_source' in st.session_state
        assert 'key_target' in st.session_state
        assert isinstance(st.session_state.mapping, dict)

def test_handle_matching_execution(mock_streamlit, sample_dataframe, mock_dask_dataframe):
    """Test matching execution with mocked Dask"""
    st.session_state.df_source = sample_dataframe
    st.session_state.df_target = sample_dataframe
    st.session_state.key_source = ['id']
    st.session_state.key_target = ['id']
    st.session_state.mapping = {
        'key_source': ['id'],
        'key_target': ['id']
    }
    
    with patch('utils.execute_matching_dask', return_value=(mock_dask_dataframe, {
        'total_match': 1,
        'missing_source': 1,
        'missing_target': 1
    })):
        handle_matching_execution()
        assert st.session_state.matching_results is not None

def test_handle_validation_rules(mock_streamlit, sample_dataframe):
    """Test validation rules handling"""
    st.session_state.matching_results = sample_dataframe
    st.session_state.df_target = sample_dataframe
    
    with patch('streamlit.expander') as mock_expander:
        mock_expander.return_value.__enter__.return_value = MagicMock()
        handle_validation_rules()
        
        assert st.session_state.validation_rules is not None
        assert isinstance(st.session_state.validation_rules, dict)
