import pytest
import pandas as pd
from state import SessionState
import streamlit as st

def test_session_state_initialization():
    """Test proper initialization of session state."""
    SessionState.initialize()
    
    assert st.session_state.step == 1
    assert st.session_state.df_source is None
    assert st.session_state.df_target is None
    assert isinstance(st.session_state.mapping, dict)
    assert isinstance(st.session_state.validation_rules, dict)
    assert isinstance(st.session_state.key_source, list)
    assert isinstance(st.session_state.key_target, list)

def test_dataframe_operations():
    """Test dataframe get/set operations."""
    test_df = pd.DataFrame({'test': [1, 2, 3]})
    
    SessionState.set_dataframe('test_df', test_df)
    retrieved_df = SessionState.get_dataframe('test_df')
    
    assert retrieved_df is not None
    assert test_df.equals(retrieved_df)
    
    # Test non-existent dataframe
    assert SessionState.get_dataframe('nonexistent') is None

def test_mapping_update():
    """Test mapping configuration updates."""
    test_mapping = {
        'key_source': ['ID'],
        'key_target': ['ID_Target'],
        'mappings': {'col1': {'destinations': ['col2']}}
    }
    
    SessionState.update_mapping(test_mapping)
    assert st.session_state.mapping == test_mapping

def test_validation_rules_update():
    """Test validation rules updates."""
    test_rules = {
        'col1': {'validate_nulls': True},
        'col2': {'validate_range': True, 'min_value': 0}
    }
    
    SessionState.update_validation_rules(test_rules)
    assert st.session_state.validation_rules == test_rules
