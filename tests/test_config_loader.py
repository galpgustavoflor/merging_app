import unittest
from unittest.mock import MagicMock, patch
import json
import streamlit as st
from utils import ConfigLoader

class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        # Clear session state before each test
        for key in list(st.session_state.keys()):
            del st.session_state[key]

    def test_load_json_config_valid(self):
        valid_json = '{"key": "value"}'
        result = ConfigLoader.load_json_config(valid_json)
        self.assertEqual(result, {"key": "value"})

    def test_load_json_config_invalid(self):
        invalid_json = '{"key": value}'  # Missing quotes
        with patch('streamlit.error') as mock_error:
            result = ConfigLoader.load_json_config(invalid_json)
            self.assertEqual(result, {})
            mock_error.assert_called_once()

    def test_load_mapping_from_json(self):
        mock_file = MagicMock()
        mock_file.read.return_value = '{"source": "target"}'
        ConfigLoader.load_mapping_from_json(mock_file)
        self.assertEqual(st.session_state.mapping, {"source": "target"})

    def test_load_mapping_from_json_invalid(self):
        mock_file = MagicMock()
        mock_file.read.return_value = 'invalid json'
        with patch('streamlit.error') as mock_error:
            ConfigLoader.load_mapping_from_json(mock_file)
            mock_error.assert_called_once()

    def test_load_validations_from_json(self):
        mock_file = MagicMock()
        mock_file.read.return_value = '{"column": {"rule": "value"}}'
        ConfigLoader.load_validations_from_json(mock_file)
        self.assertEqual(st.session_state.validation_rules, {"column": {"rule": "value"}})

    def test_load_validations_from_json_invalid(self):
        mock_file = MagicMock()
        mock_file.read.return_value = 'invalid json'
        with patch('streamlit.error') as mock_error:
            ConfigLoader.load_validations_from_json(mock_file)
            mock_error.assert_called_once()

if __name__ == '__main__':
    unittest.main()
