from typing import Optional
import pandas as pd
import streamlit as st
from .base import BaseService
from utils.file_loader import FileLoader
from constants import FILE_TYPES

class FileService(BaseService):
    def __init__(self):
        super().__init__()
        self.loader = FileLoader()

    def load_local_file(self, uploader_key: str) -> Optional[pd.DataFrame]:
        try:
            uploaded_file = st.file_uploader(
                "Load File",
                type=FILE_TYPES,
                key=uploader_key
            )
            if uploaded_file:
                self.log_operation("load_local_file", filename=uploaded_file.name)
                return self.loader.load_file(uploaded_file)
            return None
        except Exception as e:
            return self.handle_error("load_local_file", e)

    def save_file(self, df: pd.DataFrame, filename: str) -> Optional[str]:
        try:
            self.log_operation("save_file", filename=filename)
            return df.to_csv(index=False)
        except Exception as e:
            return self.handle_error("save_file", e)
