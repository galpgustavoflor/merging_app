from typing import Optional
import dask.dataframe as dd
from connections.datahub_dask_connector import ClientDask
from utils.logging_utils import get_logger

logger = get_logger(__name__)

def execute_query(client: ClientDask, query: str) -> Optional[dd.DataFrame]:
    """Execute a query using the ClientDask instance."""
    try:
        return client.query(query)
    except Exception as e:
        logger.error(f"Query execution failed: {str(e)}")
        return None

def get_table_columns(client: ClientDask, table_name: str) -> Optional[dd.DataFrame]:
    """Get column information for a specific table."""
    try:
        return client.list_columns(table_name)
    except Exception as e:
        logger.error(f"Failed to get columns for table {table_name}: {str(e)}")
        return None
