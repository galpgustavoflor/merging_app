from typing import Optional, Dict, Any
import dask.dataframe as dd
from connections.datahub_dask_connector import ClientDask
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class DataHubProcessor:
    def __init__(self, username: str, password: str):
        self.client = ClientDask(username=username, token=password)
        self.data = None

    def connect(self) -> bool:
        """Test connection to DataHub."""
        try:
            return self.client.check_host_reachable(self.client.host, self.client.port)
        except Exception as e:
            logger.error(f"Failed to connect: {str(e)}")
            return False

    def execute_query(self, query: str) -> Optional[dd.DataFrame]:
        """Execute query and store result."""
        try:
            self.data = self.client.query(query)
            return self.data
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return None

    def get_tables(self) -> Optional[dd.DataFrame]:
        """Get available tables."""
        try:
            return self.client.list_tables()
        except Exception as e:
            logger.error(f"Failed to get tables: {str(e)}")
            return None

    def get_table_columns(self, table_name: str) -> Optional[dd.DataFrame]:
        """Get columns for a specific table."""
        try:
            return self.client.list_columns(table_name)
        except Exception as e:
            logger.error(f"Failed to get columns: {str(e)}")
            return None
