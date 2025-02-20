from typing import Dict, Any
import dask.dataframe as dd
import pandas as pd
from config.settings import VALIDATION_CONFIG
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class DataValidator:
    def __init__(self, data: dd.DataFrame):
        self.data = data
        self.validation_results = {
            'null_checks': {},
            'dtype_checks': {}
        }
    
    def validate(self) -> bool:
        try:
            self._check_nulls()
            self._check_data_types()
            return self._evaluate_results()
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            return False
    
    def _check_nulls(self) -> None:
        """Enhanced null checking."""
        if hasattr(self.data, 'compute'):  # Handle dask DataFrame
            null_counts = self.data.isnull().sum().compute()
            total_rows = len(self.data)
        else:  # Handle pandas DataFrame
            null_counts = self.data.isnull().sum()
            total_rows = len(self.data)
            
        self.validation_results["null_checks"] = {
            col: {
                "count": int(count),
                "percentage": (count / total_rows) * 100,
                "status": "OK" if (count / total_rows) * 100 <= (100 - VALIDATION_CONFIG["pass_threshold"]) else "FAIL"
            } for col, count in null_counts.items()
        }
    
    def _check_data_types(self) -> None:
        """Enhanced data type checking."""
        self.validation_results["dtype_checks"] = {
            col: {
                "dtype": str(dtype),
                "numeric": pd.api.types.is_numeric_dtype(dtype),
                "temporal": pd.api.types.is_datetime64_any_dtype(dtype)
            } for col, dtype in self.data.dtypes.items()
        }
    
    def _evaluate_results(self) -> bool:
        null_percentages = [result["percentage"] for result in self.validation_results["null_checks"].values()]
        return all(p <= (100 - VALIDATION_CONFIG["pass_threshold"]) for p in null_percentages)
    
    def get_validation_report(self) -> Dict[str, Any]:
        return self.validation_results
