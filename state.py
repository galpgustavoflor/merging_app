from typing import Optional, Dict, Any, Union, List
import pandas as pd
import streamlit as st
import dask.dataframe as dd
from constants import Step
import logging
import copy
from config import DASK_CONFIG, STREAMLIT_CONFIG, VALIDATION_CONFIG

logger = logging.getLogger(__name__)

class SessionState:
    """Centralized session state management with separation of UI and data concerns."""
    
    # Configuration keys that can be updated dynamically
    CONFIG_KEYS = {
        'dask_config': DASK_CONFIG,
        'streamlit_config': STREAMLIT_CONFIG,
        'validation_config': VALIDATION_CONFIG
    }

    @staticmethod
    def initialize() -> None:
        """Initialize session state variables."""
        # UI navigation state
        if 'step' not in st.session_state:
            st.session_state.step = Step.SOURCE_UPLOAD.value
        
        # UI interaction state
        if 'show_rule_dialog' not in st.session_state:
            st.session_state.show_rule_dialog = False
        if 'previous_upload' not in st.session_state:
            st.session_state.previous_upload = None
        if 'rule_name' not in st.session_state:
            st.session_state.rule_name = ''
        if 'columns' not in st.session_state:
            st.session_state.columns = []
            
        # Data references
        if 'df_source' not in st.session_state:
            st.session_state.df_source = None
        if 'df_target' not in st.session_state:
            st.session_state.df_target = None
            
        # Configuration state
        if 'mapping' not in st.session_state:
            st.session_state.mapping = {}
        if 'validation_rules' not in st.session_state:
            st.session_state.validation_rules = {}
        if 'business_rules' not in st.session_state:
            st.session_state.business_rules = []
        
        # Temporary rule building state
        if 'current_rule' not in st.session_state:
            st.session_state.current_rule = {
                'name': '',
                'conditions': [],
                'then': []
            }
            
        # Results state (references to output from logic)
        if 'matching_results' not in st.session_state:
            st.session_state.matching_results = None
        if 'validation_results' not in st.session_state:
            st.session_state.validation_results = None
        if 'business_rule_violations' not in st.session_state:
            st.session_state.business_rule_violations = None
            
        # Dynamic configuration storage
        if 'config' not in st.session_state:
            st.session_state.config = {
                'dask_config': copy.deepcopy(DASK_CONFIG),
                'streamlit_config': copy.deepcopy(STREAMLIT_CONFIG),
                'validation_config': copy.deepcopy(VALIDATION_CONFIG)
            }

    @staticmethod
    def is_valid_dataframe(df: Union[pd.DataFrame, dd.DataFrame]) -> bool:
        """Check if input is a valid DataFrame."""
        return isinstance(df, (pd.DataFrame, dd.DataFrame))

    @staticmethod
    def set_dataframe(key: str, df: Union[pd.DataFrame, dd.DataFrame]) -> None:
        """Safely store DataFrame in session state"""
        try:
            if not SessionState.is_valid_dataframe(df):
                raise ValueError("Invalid dataframe")
            st.session_state[key] = df
            # Update columns when target DataFrame is set
            if key == 'df_target':
                st.session_state.columns = df.columns.tolist()
            logger.info(f"Successfully stored DataFrame '{key}' in session state")
            # Reset dependent states
            SessionState.reset_dependent_states(key)
        except Exception as e:
            logger.error(f"Error storing DataFrame '{key}': {str(e)}")
            raise

    @staticmethod
    def get_dataframe(key: str) -> Union[pd.DataFrame, dd.DataFrame, None]:
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
            # Reset dependent states if needed
            SessionState.reset_dependent_states(key)
        except Exception as e:
            logger.error(f"Error storing value for key '{key}': {str(e)}")
            raise

    @staticmethod
    def get_value(key: str, default: Any = None) -> Any:
        """Safely retrieve value from session state"""
        try:
            value = st.session_state.get(key, default)
            return value
        except Exception as e:
            logger.error(f"Error retrieving value for key '{key}': {str(e)}")
            return default

    @staticmethod
    def get_config(config_type: str, key: str = None, default: Any = None) -> Any:
        """Get configuration value from dynamic config storage"""
        try:
            if config_type not in st.session_state.config:
                return default
                
            if key is None:
                return st.session_state.config[config_type]
                
            return st.session_state.config[config_type].get(key, default)
        except Exception as e:
            logger.error(f"Error retrieving config '{config_type}.{key}': {str(e)}")
            return default
            
    @staticmethod
    def set_config(config_type: str, key: str, value: Any) -> None:
        """Update configuration value in dynamic config storage"""
        try:
            if config_type not in st.session_state.config:
                st.session_state.config[config_type] = {}
                
            st.session_state.config[config_type][key] = value
            logger.info(f"Updated configuration {config_type}.{key} = {value}")
        except Exception as e:
            logger.error(f"Error updating config '{config_type}.{key}': {str(e)}")

    @staticmethod
    def get_raw_value(key: str, default: Any = None) -> Any:
        """Get raw value without boolean conversion"""
        return st.session_state.get(key, default)

    @staticmethod
    def has_matching_results() -> bool:
        """Safely check if matching results exist without bool() context issues"""
        return "matching_results" in st.session_state and st.session_state.matching_results is not None

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
        """Update mapping configuration with validation"""
        if mapping is None:
            mapping = {"mappings": {}, "key_source": [], "key_target": []}
        
        # Ensure mapping has correct structure
        mapping.setdefault("mappings", {})
        mapping.setdefault("key_source", [])
        mapping.setdefault("key_target", [])
        
        # Store in session state
        st.session_state.mapping = mapping
        logger.debug("Updated mapping configuration in session state")
        # Reset dependent states
        SessionState.reset_dependent_states("mapping")

    @staticmethod
    def reset_dependent_states(trigger_key: str) -> None:
        """Reset dependent states when a key state changes"""
        resets = {
            "df_source": ["matching_results"],
            "df_target": ["matching_results", "validation_results", "business_rule_violations"],
            "mapping": ["matching_results"],
            "validation_rules": ["validation_results"],
            "business_rules": ["business_rule_violations"]
        }
        
        if trigger_key in resets:
            for key in resets[trigger_key]:
                if key in st.session_state:
                    del st.session_state[key]
                    logger.info(f"Reset {key} due to change in {trigger_key}")

    @staticmethod
    def update_validation_rules(rules: Dict) -> None:
        """Update validation rules with reset of dependent states"""
        st.session_state.validation_rules = rules
        # Reset dependent states
        SessionState.reset_dependent_states("validation_rules")

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

    @staticmethod
    def clear_dataframe(key: str) -> None:
        """Clear DataFrame from session state."""
        if key in st.session_state:
            del st.session_state[key]
            # Reset dependent states
            SessionState.reset_dependent_states(key)

    @staticmethod
    def has_dataframe(key: str) -> bool:
        """Check if DataFrame exists in session state."""
        return key in st.session_state and st.session_state[key] is not None
        
    @staticmethod
    def get_step() -> int:
        """Get current step value"""
        return st.session_state.get("step", Step.SOURCE_UPLOAD.value)
        
    @staticmethod
    def next_step() -> None:
        """Advance to next step"""
        st.session_state.step = st.session_state.get("step", Step.SOURCE_UPLOAD.value) + 1
        
    @staticmethod  
    def prev_step() -> None:
        """Go back to previous step"""
        current = st.session_state.get("step", Step.SOURCE_UPLOAD.value)
        if current > Step.SOURCE_UPLOAD.value:
            st.session_state.step = current - 1
            
    @staticmethod
    def go_to_step(step: int) -> None:
        """Go to specific step"""
        st.session_state.step = step

    @staticmethod
    def has_value(key: str) -> bool:
        """Safely check if a value exists and is not None in session state"""
        return key in st.session_state and st.session_state[key] is not None
