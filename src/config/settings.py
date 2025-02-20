from pathlib import Path
from typing import Dict, Any
import os
from dotenv import load_dotenv

# Load environment variables (optional)
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Default Configuration
DEFAULT_CONFIG = {
    # Application settings
    "DEBUG": False,
    "LOG_LEVEL": "INFO",
    
    # Dask settings
    "DASK_PARTITIONS": 10,
    "SAMPLE_SIZE": 500,
    
    # DataHub settings
    "DATAHUB_HOST": "datahub.ulysses.galpenergia.corp",
    "DATAHUB_PORT": 32010,
    "DATAHUB_TIMEOUT": 5.0,
    
    # Validation settings
    "VALIDATION_THRESHOLD": 95.0,
    "MIN_SAMPLE_SIZE": 1000,
    
    # File handling settings
    "MAX_FILE_SIZE": 500,
    "CHUNK_SIZE": 10000,
}

# Application settings
APP_CONFIG: Dict[str, Any] = {
    "name": "File Mapping Process",
    "version": "1.0.0",
    "debug": os.getenv("DEBUG", str(DEFAULT_CONFIG["DEBUG"])).lower() == "true"
}

# Dask configuration
DASK_CONFIG: Dict[str, Any] = {
    "default_npartitions": int(os.getenv("DASK_PARTITIONS", DEFAULT_CONFIG["DASK_PARTITIONS"])),
    "sample_size": int(os.getenv("SAMPLE_SIZE", DEFAULT_CONFIG["SAMPLE_SIZE"]))
}

# DataHub configuration
DATAHUB_CONFIG: Dict[str, Any] = {
    "host": os.getenv("DATAHUB_HOST", DEFAULT_CONFIG["DATAHUB_HOST"]),
    "port": int(os.getenv("DATAHUB_PORT", DEFAULT_CONFIG["DATAHUB_PORT"])),
    "cache_dir": "./cache",
    "default_npartitions": 4,
    "timeout": float(os.getenv("DATAHUB_TIMEOUT", DEFAULT_CONFIG["DATAHUB_TIMEOUT"]))
}

# Validation settings
VALIDATION_CONFIG: Dict[str, Any] = {
    "pass_threshold": float(os.getenv("VALIDATION_THRESHOLD", DEFAULT_CONFIG["VALIDATION_THRESHOLD"])),
    "min_sample_size": int(os.getenv("MIN_SAMPLE_SIZE", DEFAULT_CONFIG["MIN_SAMPLE_SIZE"]))
}

# File handling settings
FILE_CONFIG: Dict[str, Any] = {
    "allowed_extensions": ["csv", "xlsx"],
    "max_file_size_mb": int(os.getenv("MAX_FILE_SIZE", DEFAULT_CONFIG["MAX_FILE_SIZE"])),
    "chunk_size": int(os.getenv("CHUNK_SIZE", DEFAULT_CONFIG["CHUNK_SIZE"]))
}

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "default": {
            "level": os.getenv("LOG_LEVEL", DEFAULT_CONFIG["LOG_LEVEL"]),
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
        "file": {
            "level": os.getenv("LOG_LEVEL", DEFAULT_CONFIG["LOG_LEVEL"]),
            "formatter": "standard",
            "class": "logging.FileHandler",
            "filename": os.path.join(BASE_DIR, "logs", "app.log"),
            "mode": "a",
        }
    },
    "loggers": {
        "": {
            "handlers": ["default", "file"],
            "level": os.getenv("LOG_LEVEL", DEFAULT_CONFIG["LOG_LEVEL"]),
            "propagate": True
        }
    }
}
