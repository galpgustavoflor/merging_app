import os
import sys
import pytest
import pandas as pd
import streamlit as st
from pathlib import Path
from unittest.mock import MagicMock

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.append(str(Path(__file__).parent.parent))

from constants import Step

@pytest.fixture(autouse=True)
def add_imports(doctest_namespace):
    """Make imports available in doctests."""
    import pandas as pd
    import numpy as np
    doctest_namespace["pd"] = pd
    doctest_namespace["np"] = np

@pytest.fixture(autouse=True)
def setup_streamlit():
    """Initialize streamlit session state before each test"""
    # Clear existing session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Initialize required session state variables
    initial_state = {
        'step': Step.SOURCE_UPLOAD.value,
        'df_source': None,
        'df_target': None,
        'mapping': {},
        'key_source': [],
        'key_target': [],
        'validation_rules': {},
        'matching_results': None,
        'validation_results': None
    }
    
    for key, value in initial_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['A', 'B', 'C'],
        'value': [10.0, 20.0, 30.0]
    })

@pytest.fixture
def mock_file():
    """Create a mock file for testing"""
    mock = MagicMock()
    mock.name = "test.csv"
    mock.size = 1024
    mock.type = "text/csv"
    return mock

@pytest.fixture
def mock_streamlit(monkeypatch):
    """Mock common streamlit functions"""
    def mock_function(*args, **kwargs):
        return None
    
    monkeypatch.setattr('streamlit.markdown', mock_function)
    monkeypatch.setattr('streamlit.write', mock_function)
    monkeypatch.setattr('streamlit.error', mock_function)
    monkeypatch.setattr('streamlit.success', mock_function)
    monkeypatch.setattr('streamlit.info', mock_function)
    monkeypatch.setattr('streamlit.dataframe', mock_function)
    monkeypatch.setattr('streamlit.download_button', mock_function)
    monkeypatch.setattr('streamlit.button', lambda *a, **kw: False)
    monkeypatch.setattr('streamlit.checkbox', lambda *a, **kw: True)
    monkeypatch.setattr('streamlit.selectbox', lambda *a, **kw: a[1][0] if len(a) > 1 else None)
    monkeypatch.setattr('streamlit.multiselect', lambda *a, **kw: a[1][:1] if len(a) > 1 else [])
    monkeypatch.setattr('streamlit.radio', lambda *a, **kw: a[1][0] if len(a) > 1 else None)

@pytest.fixture
def mock_dask_dataframe():
    """Create a mock Dask DataFrame"""
    def mock_compute():
        return pd.DataFrame({
            'id': [1, 2, 3],
            '_merge': ['both', 'left_only', 'right_only']
        })
    
    mock_ddf = MagicMock()
    mock_ddf.compute = mock_compute
    mock_ddf.merge.return_value = mock_ddf
    mock_ddf.__getitem__.return_value = mock_ddf
    mock_ddf.shape.__getitem__.return_value.compute.return_value = 3
    return mock_ddf

@pytest.fixture
def setup_environment(monkeypatch):
    """Setup environment variables for testing"""
    env_vars = {
        'DASK_PARTITIONS': '20',
        'MAX_FILE_SIZE_MB': '1000',
        'CORS_ORIGINS': 'domain1.com,domain2.com'
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
