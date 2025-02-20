import os
from typing import Dict, Any

def get_env_var(key: str, default: Any = None) -> Any:
    """Get environment variable with fallback to default"""
    return os.environ.get(key, default)

# Environment-specific configurations
ENVIRONMENT = get_env_var('APP_ENV', 'development')

DASK_CONFIG: Dict[str, Any] = {
    "default_npartitions": int(get_env_var('DASK_PARTITIONS', 10)),
    "npartitions": int(get_env_var('DASK_PARTITIONS', 10)),
    "sample_size": int(get_env_var('SAMPLE_SIZE', 1000)),
    "scheduler": get_env_var('DASK_SCHEDULER', 'threads'),
}

SECURITY_CONFIG: Dict[str, Any] = {
    "allowed_file_types": ["csv", "xlsx"],
    "max_file_size_mb": int(get_env_var('MAX_FILE_SIZE_MB', 500)),
    "content_security_policy": "default-src 'self'; script-src 'self' 'unsafe-inline';",
    "cors_origins": get_env_var('CORS_ORIGINS', '*').split(','),
}

STREAMLIT_CONFIG: Dict[str, Any] = {
    "page_title": "Dataset Comparison and Validation Tool",
    "file_types": ["csv", "xlsx"],
    "max_file_size": 500  # MB
}

VALIDATION_CONFIG: Dict[str, Any] = {
    "pass_threshold": 95.0,  # percentage
    "min_sample_size": 1000
}

DATE_FORMATS: Dict[str, str] = {
    "default": "%Y-%m-%d",
    "alternative": ["%d/%m/%Y", "%Y/%m/%d"]
}

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
            "level": "INFO",
            "formatter": "standard",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": ["default"],
            "level": "INFO",
            "propagate": True
        }
    }
}
