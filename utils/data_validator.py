import pandas as pd
import streamlit as st
from typing import List, Dict
from constants import ValidationRule as VRule

class DataValidator:
    @staticmethod
    def execute_validation(df: pd.DataFrame, validation_rules: dict) -> List[dict]:
        st.write("Executing data validation...")
        results = []
        total_records = len(df)
        
        for col, rules in validation_rules.items():
            if col not in df.columns:
                continue
                
            # Validation logic from the original utils.py
            # ... existing validation code ...

        return results
