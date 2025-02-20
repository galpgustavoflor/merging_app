import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import sys
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils import FileLoader, clean_dataframe_for_display, execute_matching_dask, handle_large_file

class TestFileLoader(unittest.TestCase):
    def setUp(self):
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        
        self.mock_file = MagicMock()
        self.mock_file.name = "test.csv"
        self.mock_file.size = 1024
        self.mock_file.type = "text/csv"

    @patch('pandas.read_csv')
    def test_load_file_csv(self, mock_read_csv):
        test_data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        mock_read_csv.return_value = test_data.copy()
        
        with patch('utils.FileLoader._process_dataframe', return_value=test_data):
            result = FileLoader.load_file(self.mock_file)
            pd.testing.assert_frame_equal(result, test_data)

    def test_load_file_size_limit(self):
        self.mock_file.size = 1024 * 1024 * 1024  # 1GB
        with patch('logging.getLogger') as mock_logger:
            with self.assertRaises(ValueError):
                FileLoader.load_file(self.mock_file)

    def test_process_dataframe(self):
        test_df = pd.DataFrame({
            'col1': ['1', '2', '3'],
            'col2': ['a', 'b', 'c'],
            'col3': ['\ufeff4', '5', '6']
        })
        result = FileLoader._process_dataframe(test_df)
        self.assertEqual(result['col1'].dtype.name, 'int64')
        self.assertEqual(result['col2'].dtype.name, 'object')
        self.assertEqual(list(result['col3']), ['4', '5', '6'])

    def test_clean_dataframe_for_display(self):
        df = pd.DataFrame({
            'int_col': pd.array([1, 2, None], dtype='Int64'),
            'float_col': [1.1, None, 3.3],
            'str_col': ['a', 'b', None],
            'date_col': pd.date_range('2023-01-01', periods=3)
        })
        result = clean_dataframe_for_display(df)
        self.assertTrue(all(isinstance(x, (int, type(None))) for x in result['int_col']))
        self.assertTrue(all(isinstance(x, str) for x in result['str_col'].dropna()))

    @patch('pandas.read_csv')
    def test_handle_large_file(self, mock_read_csv):
        test_chunks = [
            pd.DataFrame({'col1': [1, 2, 3]}),
            pd.DataFrame({'col1': [4, 5, 6]})
        ]
        mock_read_csv.return_value = pd.concat(test_chunks)
        
        result = handle_large_file('test.csv', chunk_size=1000)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 6)

    @patch('dask.dataframe.from_pandas')
    def test_execute_matching_dask(self, mock_from_pandas):
        df_source = pd.DataFrame({
            'key': [1, 2, 3],
            'value': ['a', 'b', 'c']
        })
        df_target = pd.DataFrame({
            'key': [2, 3, 4],
            'other': ['x', 'y', 'z']
        })
        
        mock_ddf = MagicMock()
        mock_ddf.merge.return_value = mock_ddf
        mock_ddf.__getitem__.return_value = mock_ddf
        mock_ddf.shape.__getitem__.return_value.compute.return_value = 2
        mock_from_pandas.return_value = mock_ddf
        
        result, stats = execute_matching_dask(df_source, df_target, ['key'], ['key'])
        self.assertEqual(stats['total_match'], 2)
