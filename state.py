from typing import Optional, Dict, List
import pandas as pd
import streamlit as st

class SessionState:
    @staticmethod
    def initialize() -> None:
        """Initialize all session state variables."""
        if 'step' not in st.session_state:
            st.session_state.step = 1
        
        defaults = {
            'df_source': None,
            'df_target': None,
            'mapping': {},
            'validation_rules': {},
            'key_source': [],
            'key_target': [],
            'show_json': False,
            'update_ui': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def get_dataframe(key: str) -> Optional[pd.DataFrame]:
        """Get a dataframe from session state."""
        return st.session_state.get(key)

    @staticmethod
    def set_dataframe(key: str, df: pd.DataFrame) -> None:
        """Set a dataframe in session state."""
        st.session_state[key] = df

    @staticmethod
    def update_mapping(mapping: Dict) -> None:
        """Update mapping configuration."""
        st.session_state.mapping = mapping

    @staticmethod
    def update_validation_rules(rules: Dict) -> None:
        """Update validation rules."""
        st.session_state.validation_rules = rules
