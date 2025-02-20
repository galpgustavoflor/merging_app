from abc import ABC
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class BaseService(ABC):
    def __init__(self):
        self.logger = logger

    def handle_error(self, operation: str, error: Exception) -> Optional[Any]:
        self.logger.error(f"Error in {operation}: {str(error)}", exc_info=True)
        return None

    def log_operation(self, operation: str, **kwargs) -> None:
        self.logger.info(f"Executing {operation}", extra=kwargs)
