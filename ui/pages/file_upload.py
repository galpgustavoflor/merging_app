from typing import Optional, Union
import streamlit as st
import pandas as pd
from .base import BasePage
from services.file_service import FileService
from services.datahub_service import DataHubService
from ui.components import DataDisplay

class FileUploadPage(BasePage):
    def __init__(self, file_type: str):
        super().__init__(f"Load {file_type.capitalize()} File")
        self.file_type = file_type
        self.file_service = FileService()
        self.datahub_service = DataHubService()

    def render(self) -> None:
        st.header(f"Step {1 if self.file_type == 'source' else 2}: {self.title}")
        
        source = st.radio(
            "Select Source File Type",
            ["Local File", "DataHub"],
            key=f"{self.file_type}_source_type"
        )
        
        df = None
        if source == "Local File":
            df = self._handle_local_file()
        else:
            df = self._handle_datahub()
            
        if df is not None:
            self._display_and_proceed(df)

    def _handle_local_file(self) -> Optional[pd.DataFrame]:
        return self.file_service.load_local_file(f"{self.file_type}_uploader")

    def _handle_datahub(self) -> Optional[pd.DataFrame]:
        return self.datahub_service.load_file(self.file_type)

    def _display_and_proceed(self, df: pd.DataFrame) -> None:
        self.session.set_dataframe(f'df_{self.file_type}', df)
        DataDisplay.show_metadata(df, f"{self.file_type.capitalize()} Data")
        self.next_step()
