import unittest
from unittest.mock import patch
import os
from config import (
    get_env_var, DASK_CONFIG, SECURITY_CONFIG, 
    STREAMLIT_CONFIG, VALIDATION_CONFIG
)

class TestConfig(unittest.TestCase):
    def setUp(self):
        # Clear any existing environment variables
        if 'DASK_PARTITIONS' in os.environ:
            del os.environ['DASK_PARTITIONS']

    def test_get_env_var_with_default(self):
        result = get_env_var('NONEXISTENT_VAR', 'default')
        self.assertEqual(result, 'default')

    def test_get_env_var_from_environment(self):
        os.environ['TEST_VAR'] = 'test_value'
        result = get_env_var('TEST_VAR', 'default')
        self.assertEqual(result, 'test_value')
        del os.environ['TEST_VAR']

    def test_dask_config_defaults(self):
        self.assertIn('default_npartitions', DASK_CONFIG)
        self.assertIn('sample_size', DASK_CONFIG)
        self.assertIn('scheduler', DASK_CONFIG)

    def test_security_config_defaults(self):
        self.assertIn('allowed_file_types', SECURITY_CONFIG)
        self.assertIn('max_file_size_mb', SECURITY_CONFIG)
        self.assertIn('content_security_policy', SECURITY_CONFIG)
        self.assertIn('cors_origins', SECURITY_CONFIG)

    def test_streamlit_config_defaults(self):
        self.assertIn('page_title', STREAMLIT_CONFIG)
        self.assertIn('file_types', STREAMLIT_CONFIG)
        self.assertIn('max_file_size', STREAMLIT_CONFIG)

    def test_validation_config_defaults(self):
        self.assertIn('pass_threshold', VALIDATION_CONFIG)
        self.assertIn('min_sample_size', VALIDATION_CONFIG)

    @patch.dict(os.environ, {'DASK_PARTITIONS': '20'})
    def test_dask_config_from_env(self):
        from config import DASK_CONFIG
        self.assertEqual(DASK_CONFIG['default_npartitions'], 20)

    @patch.dict(os.environ, {'MAX_FILE_SIZE_MB': '1000'})
    def test_security_config_from_env(self):
        from config import SECURITY_CONFIG
        self.assertEqual(SECURITY_CONFIG['max_file_size_mb'], 1000)

    @patch.dict(os.environ, {'CORS_ORIGINS': 'domain1.com,domain2.com'})
    def test_security_config_cors_origins(self):
        from config import SECURITY_CONFIG
        self.assertEqual(SECURITY_CONFIG['cors_origins'], ['domain1.com', 'domain2.com'])

if __name__ == '__main__':
    unittest.main()
