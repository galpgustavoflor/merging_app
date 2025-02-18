from typing import Optional, Dict
import pandas as pd
import streamlit as st

class SessionState:
    @staticmethod
    def initialize() -> None:
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
            'update_ui': False,
            'matching_results': None,
            'validation_results': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    @staticmethod
    def get_dataframe(key: str) -> Optional[pd.DataFrame]:
        return st.session_state.get(key)

    @staticmethod
    def set_dataframe(key: str, df: pd.DataFrame) -> None:
        st.session_state[key] = df

    @staticmethod
    def update_mapping(mapping: Dict) -> None:
        st.session_state.mapping = mapping

    @staticmethod
    def update_validation_rules(rules: Dict) -> None:
        st.session_state.validation_rules = rules
