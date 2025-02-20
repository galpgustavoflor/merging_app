import pandas as pd
import numpy as np
import dask.dataframe as dd
import io
from typing import Dict, Any, Union, Optional
from ydata_profiling import ProfileReport
from config.settings import DASK_CONFIG, FILE_CONFIG, DATAHUB_CONFIG
import requests
from utils.logging_utils import get_logger
import pyarrow as pa

logger = get_logger(__name__)

class FileProcessor:
    def __init__(self, file_obj=None):
        self.file_obj = file_obj
        self.data = None
        if file_obj is not None:
            self.load_file()
    
    def load_file(self) -> None:
        """Load data from uploaded file."""
        try:
            if self.file_obj is None:
                raise ValueError("No file object provided")

            file_buffer = io.BytesIO(self.file_obj.getvalue())
            file_extension = self.file_obj.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # Try to infer data types and handle mixed types
                df = pd.read_csv(
                    file_buffer,
                    low_memory=False,
                    dtype_backend='pyarrow',  # Use PyArrow backend
                    on_bad_lines='warn'  # Don't fail on problematic lines
                )
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(file_buffer)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            if df.empty:
                raise ValueError("The file contains no data")
            
            if len(df.columns) == 0:
                raise ValueError("No columns detected in the file")
                
            # Clean and standardize the data
            df = self._standardize_dataframe(df)
            
            self.data = df
            logger.info(f"Successfully loaded file: {self.file_obj.name}")
            
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            raise

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize DataFrame for compatibility."""
        try:
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Handle problematic data types
            for col in df.columns:
                try:
                    # Convert integer types to float64
                    if pd.api.types.is_integer_dtype(df[col]):
                        df[col] = df[col].astype('float64')
                    # Convert object types with mainly numbers to float
                    elif pd.api.types.is_object_dtype(df[col]):
                        if df[col].nunique() / len(df[col]) < 0.5:  # Less than 50% unique values
                            try:
                                df[col] = pd.to_numeric(df[col], errors='ignore')
                            except:
                                pass
                except Exception as e:
                    logger.warning(f"Could not convert column {col}: {str(e)}")
                    continue
            
            return df
            
        except Exception as e:
            logger.error(f"Error standardizing DataFrame: {str(e)}")
            raise

    def generate_profile(self) -> str:
        """Generate a profile report and return HTML."""
        try:
            if isinstance(self.data, dd.DataFrame):
                profile_df = self.data.head(DASK_CONFIG["sample_size"]).compute()
            else:
                profile_df = self.data.head(DASK_CONFIG["sample_size"])

            profile = ProfileReport(
                profile_df,
                title="Data Profile Report",
                minimal=True,
                explorative=True,
                progress_bar=False
            )
            return profile.to_html()
        except Exception as e:
            logger.error(f"Error generating profile: {str(e)}")
            raise

    def get_summary(self) -> Dict[str, Any]:
        """Get enhanced summary statistics."""
        try:
            if self.data is None:
                raise ValueError("No data loaded")
            
            if self.data.empty:
                raise ValueError("DataFrame is empty")
            
            if len(self.data.columns) == 0:
                raise ValueError("DataFrame has no columns")

            # Get total rows and sample data
            total_rows = len(self.data)
            computed_data = self.data.head(DASK_CONFIG["sample_size"])
            
            # Calculate memory usage safely
            try:
                memory_usage = computed_data.memory_usage(deep=True)
                total_memory = float(memory_usage.sum()) / 1024 / 1024  # MB
            except Exception as e:
                logger.warning(f"Could not calculate memory usage: {str(e)}")
                total_memory = 0
                memory_usage = pd.Series([0] * len(computed_data.columns), index=computed_data.columns)

            # Get numeric summary safely
            try:
                numeric_cols = computed_data.select_dtypes(include=[np.number]).columns
                numeric_summary = computed_data[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
            except Exception as e:
                logger.warning(f"Could not generate numeric summary: {str(e)}")
                numeric_summary = {}

            return {
                "columns": list(computed_data.columns),
                "shape": (total_rows, len(computed_data.columns)),  # Use total rows instead of sample size
                "sample_shape": computed_data.shape,  # Add sample shape separately
                "dtypes": {col: str(dtype) for col, dtype in computed_data.dtypes.items()},
                "sample": computed_data,
                "numeric_summary": numeric_summary,
                "memory_usage": total_memory,
                "column_memory": dict(memory_usage / 1024),
                "null_counts": computed_data.isnull().sum().to_dict(),
                "unique_counts": {col: computed_data[col].nunique() 
                                for col in computed_data.columns}
            }
        except Exception as e:
            logger.error(f"Error getting summary: {str(e)}")
            raise
