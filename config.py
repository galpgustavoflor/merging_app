from typing import Dict, Any

DASK_CONFIG: Dict[str, Any] = {
    "default_npartitions": 10,
    "sample_size": 500
}

STREAMLIT_CONFIG: Dict[str, Any] = {
    "page_title": "File Mapping Process",
    "file_types": ["csv", "xlsx"],
    "max_file_size": 500  # MB
}

VALIDATION_CONFIG: Dict[str, Any] = {
    "pass_threshold": 50,  # percentage
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
