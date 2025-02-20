from typing import Optional, Dict, Any
import pandas as pd
import streamlit as st
from constants import Step
import logging

logger = logging.getLogger(__name__)

class SessionState:
    """Centralized session state management."""

    @staticmethod
    def initialize() -> None:
        if 'step' not in st.session_state:
            st.session_state.step = Step.SOURCE_UPLOAD.value
        if 'df_source' not in st.session_state:
            st.session_state.df_source = None
        if 'df_target' not in st.session_state:
            st.session_state.df_target = None
        if 'mapping' not in st.session_state:
            st.session_state.mapping = {}
        if 'validation_rules' not in st.session_state:
            st.session_state.validation_rules = {}
        if 'matching_results' not in st.session_state:
            st.session_state.matching_results = None
        if 'validation_results' not in st.session_state:
            st.session_state.validation_results = None

    @staticmethod
    def set_dataframe(key: str, df: pd.DataFrame) -> None:
        """Safely store DataFrame in session state"""
        try:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Invalid dataframe")
            st.session_state[key] = df
            logger.info(f"Successfully stored DataFrame '{key}' in session state")
        except Exception as e:
            logger.error(f"Error storing DataFrame '{key}': {str(e)}")
            raise

    @staticmethod
    def get_dataframe(key: str) -> Optional[pd.DataFrame]:
        """Safely retrieve DataFrame from session state"""
        try:
            if key not in st.session_state:
                raise KeyError(f"{key} not found")
            return st.session_state.get(key)
        except Exception as e:
            logger.error(f"Error retrieving DataFrame '{key}': {str(e)}")
            return None

    @staticmethod
    def set_value(key: str, value: Any) -> None:
        """Safely store value in session state"""
        try:
            st.session_state[key] = value
            logger.debug(f"Stored value for key '{key}' in session state")
        except Exception as e:
            logger.error(f"Error storing value for key '{key}': {str(e)}")
            raise

    @staticmethod
    def get_value(key: str, default: Any = None) -> Any:
        """Safely retrieve value from session state"""
        try:
            return st.session_state.get(key, default)
        except Exception as e:
            logger.error(f"Error retrieving value for key '{key}': {str(e)}")
            return default

    @staticmethod
    def clear() -> None:
        """Clear all session state data"""
        try:
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            logger.info("Successfully cleared session state")
        except Exception as e:
            logger.error(f"Error clearing session state: {str(e)}")
            raise

    @staticmethod
    def update_mapping(mapping: Dict) -> None:
        st.session_state.mapping = mapping

    @staticmethod
    def update_validation_rules(rules: Dict) -> None:
        st.session_state.validation_rules = rules

    @staticmethod
    def clear_state(key: str) -> None:
        """Clear specific key from session state"""
        try:
            if key in st.session_state:
                del st.session_state[key]
            else:
                raise KeyError(f"{key} not found in session state")
        except Exception as e:
            logger.error(f"Error clearing key '{key}' from session state: {str(e)}")
            raise
