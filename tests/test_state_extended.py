import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from state import SessionState

class TestStateExtended(unittest.TestCase):
    def setUp(self):
        # Clear session state before each test
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        self.test_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })

    def test_set_dataframe_success(self):
        with patch('logging.getLogger') as mock_logger:
            SessionState.set_dataframe('test_df', self.test_df)
            self.assertTrue(mock_logger.return_value.info.called)
            pd.testing.assert_frame_equal(st.session_state['test_df'], self.test_df)

    def test_set_dataframe_error(self):
        with patch('logging.getLogger') as mock_logger:
            with self.assertRaises(Exception):
                SessionState.set_dataframe('test_df', "not a dataframe")
            self.assertTrue(mock_logger.return_value.error.called)

    def test_get_dataframe_nonexistent(self):
        with patch('logging.getLogger') as mock_logger:
            result = SessionState.get_dataframe('nonexistent')
            self.assertIsNone(result)
            self.assertTrue(mock_logger.return_value.error.called)

    def test_set_value_different_types(self):
        test_values = [
            ('string', "test"),
            ('integer', 42),
            ('float', 3.14),
            ('list', [1, 2, 3]),
            ('dict', {'key': 'value'}),
            ('none', None)
        ]
        
        for key, value in test_values:
            SessionState.set_value(key, value)
            self.assertEqual(st.session_state[key], value)

    def test_get_value_with_default(self):
        default_value = "default"
        result = SessionState.get_value('nonexistent', default_value)
        self.assertEqual(result, default_value)

    def test_clear_state_with_data(self):
        # Set up some test data
        SessionState.set_value('test1', 'value1')
        SessionState.set_value('test2', 'value2')
        SessionState.set_dataframe('df', self.test_df)
        
        # Clear the state
        SessionState.clear()
        
        # Verify state is empty
        self.assertEqual(len(st.session_state), 0)

    def test_clear_state_error_handling(self):
        with patch('streamlit.session_state') as mock_state:
            mock_state.__delitem__.side_effect = Exception("Test error")
            with patch('logging.getLogger') as mock_logger:
                with self.assertRaises(Exception):
                    SessionState.clear()
                self.assertTrue(mock_logger.return_value.error.called)

    def test_set_get_value_consistency(self):
        test_value = {'complex': [1, 2, {'nested': 'value'}]}
        SessionState.set_value('complex_value', test_value)
        retrieved_value = SessionState.get_value('complex_value')
        self.assertEqual(retrieved_value, test_value)

    def test_state_persistence(self):
        # Test that values persist in session state
        SessionState.set_value('persistent', 'value')
        self.assertEqual(SessionState.get_value('persistent'), 'value')
        
        # Modify value and check persistence
        SessionState.set_value('persistent', 'new_value')
        self.assertEqual(SessionState.get_value('persistent'), 'new_value')

if __name__ == '__main__':
    unittest.main()
