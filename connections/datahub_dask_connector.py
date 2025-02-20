import dask.dataframe as dd
import logging
from typing import Optional
import time
import socket  # Adicionar esta linha
from .pyarrow_client import connect_to_dremio_flight_server_endpoint

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

class ClientDask:
    def __init__(self, username: str, token: str):
        self.username = username
        self.token = token
        self.host = "datahub.ulysses.galpenergia.corp"
        self.port = 32010
        logger.info(f"Initializing ClientDask for user: {username}")

    @staticmethod
    def check_host_reachable(host: str, port: int, timeout: float = 5.0) -> bool:
        logger.debug(f"Checking host reachability: {host}:{port}")
        start_time = time.time()
        try:
            socket.create_connection((host, port), timeout=timeout)
            elapsed = time.time() - start_time
            logger.info(f"Host {host}:{port} is reachable (took {elapsed:.2f}s)")
            return True
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Host {host}:{port} is not reachable after {elapsed:.2f}s: {str(e)}")
            return False

    def query(self, query: str) -> Optional[dd.DataFrame]:
        """Execute a query with simple error handling"""
        try:
            dask_df = connect_to_dremio_flight_server_endpoint(
                self.host,
                self.port,
                self.username,
                self.token,
                query,
                True, 
                False, 
                True, 
                False, 
                False, 
                False
            )
            return dask_df
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise

    def list_tables(self) -> dd.DataFrame:
        """
        The function to list all tables in the Dremio server.
        
        Returns:
        dd.DataFrame: The list of tables.
        """
        try:
            query = """
            SELECT 
                TABLE_SCHEMA,
                TABLE_NAME,
                TABLE_TYPE
            FROM 
                INFORMATION_SCHEMA."TABLES"
            """
            return self.query(query)
            
        except Exception as e:
            logger.error(f"Failed to list tables: {str(e)}")
            raise

    def list_columns(self, table):
        """
        The function to list all columns in a table.
        
        Parameters:
        table (str): The name of the table.
        
        Returns:
        pandas.DataFrame: The list of columns in the table.
        """
        query = f"SELECT * FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='{table}'"
        return self.query(query)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
