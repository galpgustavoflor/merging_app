import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import streamlit as st
import logging
from state import SessionState

class TestSessionState(unittest.TestCase):
    def setUp(self):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        self.test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })

    def test_set_get_dataframe_success(self):
        with patch('logging.getLogger') as mock_logger:
            SessionState.set_dataframe('test_df', self.test_df)
            self.assertTrue(mock_logger.return_value.info.called)
            pd.testing.assert_frame_equal(st.session_state['test_df'], self.test_df)

    def test_set_get_invalid_dataframe(self):
        with patch('logging.getLogger') as mock_logger:
            with self.assertRaises(Exception):
                SessionState.set_dataframe('test_df', "not a dataframe")
            self.assertTrue(mock_logger.return_value.error.called)

    def test_get_dataframe_nonexistent(self):
        with patch('logging.getLogger') as mock_logger:
            result = SessionState.get_dataframe('nonexistent')
            self.assertIsNone(result)
            self.assertTrue(mock_logger.return_value.error.called)

    def test_clear_state_error(self):
        with patch('logging.getLogger') as mock_logger:
            with patch.dict(st.session_state, {'key': 'value'}, clear=True):
                with patch.object(st.session_state, '__delitem__', side_effect=Exception("Test error")):
                    with self.assertRaises(Exception):
                        SessionState.clear()
                    self.assertTrue(mock_logger.return_value.error.called)
