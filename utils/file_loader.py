import logging
import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)

class FileLoader:
    @staticmethod
    @st.cache_data
    def load_file(uploaded_file) -> Optional[pd.DataFrame]:
        try:
            file_path = Path(uploaded_file.name)
            if file_path.suffix == '.xlsx':
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_path.suffix == '.csv':
                try:
                    df = pd.read_csv(
                        uploaded_file,
                        engine='c',
                        on_bad_lines='warn',
                        low_memory=False
                    )
                except Exception:
                    df = pd.read_csv(
                        uploaded_file,
                        engine='python',
                        on_bad_lines='skip'
                    )
            else:
                raise ValueError("Unsupported file format")
            return FileLoader._process_dataframe(df)
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}", exc_info=True)
            st.error(f"Error loading file: {str(e)}")
            return None

    @staticmethod
    def _process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.convert_dtypes()
            df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
            for col in df.columns:
                if pd.api.types.is_object_dtype(df[col]):
                    numeric_col = pd.to_numeric(df[col], errors='ignore')
                    if not pd.api.types.is_object_dtype(numeric_col):
                        df[col] = numeric_col
            return df
        except Exception as e:
            logger.error(f"Error processing dataframe: {str(e)}", exc_info=True)
            raise
