import logging
import logging.config
from config import LOGGING_CONFIG

def setup_logging():
    """Configure logging for the application."""
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.debug("Logging configured successfully")
