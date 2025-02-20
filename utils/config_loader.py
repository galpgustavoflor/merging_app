import json
import streamlit as st
from typing import Dict

class ConfigLoader:
    @staticmethod
    def load_json_config(file_content: str) -> Dict:
        try:
            return json.loads(file_content)
        except json.JSONDecodeError as e:
            st.error(f"Invalid JSON configuration: {str(e)}")
            return {}

    @staticmethod
    def load_mapping_from_json(json_file) -> None:
        try:
            content = json_file.read()
            loaded_mapping = json.loads(content)
            st.session_state.mapping = loaded_mapping
            st.success("Mapping rules loaded successfully.")
        except Exception as e:
            st.error(f"Error loading mapping rules: {str(e)}")

    @staticmethod
    def load_validations_from_json(json_file) -> None:
        try:
            content = json_file.read()
            loaded_rules = json.loads(content)
            st.session_state.validation_rules = loaded_rules
            st.success("Validation rules loaded successfully.")
        except Exception as e:
            st.error(f"Error loading validation rules: {str(e)}")
            loaded_rules = {}
        st.write("Loaded rules:", loaded_rules)
