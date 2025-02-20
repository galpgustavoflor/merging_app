from abc import ABC, abstractmethod
from typing import Any, Dict, List
import pandas as pd

class BaseValidator(ABC):
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        pass

    def check_required_columns(self, data: pd.DataFrame, required_columns: List[str]) -> bool:
        missing_columns = set(required_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        return True
