import logging
import logging.config
import os
from pathlib import Path
from config.settings import LOGGING_CONFIG, BASE_DIR

def ensure_log_directory() -> None:
    """Ensure the logs directory exists."""
    log_dir = Path(BASE_DIR) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

def setup_logging() -> None:
    """Configure logging based on settings."""
    try:
        # Ensure logs directory exists
        ensure_log_directory()
        
        # Create base log file if it doesn't exist
        log_file = Path(BASE_DIR) / "logs" / "app.log"
        if not log_file.exists():
            log_file.touch()
            
        logging.config.dictConfig(LOGGING_CONFIG)
        logger = logging.getLogger(__name__)
        logger.info("Logging setup completed successfully")
    except Exception as e:
        # Fallback to basic logging if configuration fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[logging.StreamHandler()]
        )
        logger = logging.getLogger(__name__)
        logger.warning(f"Failed to setup custom logging. Using basic configuration. Error: {str(e)}")

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name."""
    return logging.getLogger(name)
