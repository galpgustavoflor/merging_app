import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import dask.dataframe as dd
from utils import (
    FileLoader, clean_dataframe_for_display, 
    execute_matching_dask, handle_large_file
)

class TestUtilsExtended(unittest.TestCase):
    def setUp(self):
        self.test_df = pd.DataFrame({
            'int_col': pd.array([1, 2, None], dtype='Int64'),
            'float_col': [1.1, None, 3.3],
            'str_col': ['a', 'b', None],
            'date_col': pd.date_range('2023-01-01', periods=3)
        })

    def test_clean_dataframe_complex_types(self):
        result = clean_dataframe_for_display(self.test_df)
        self.assertTrue(all(isinstance(x, (int, type(None))) for x in result['int_col']))
        self.assertTrue(all(isinstance(x, str) for x in result['str_col'] if x is not None))

    def test_clean_dataframe_date_handling(self):
        result = clean_dataframe_for_display(self.test_df)
        self.assertTrue(all(isinstance(x, str) for x in result['date_col']))

    @patch('dask.dataframe.from_pandas')
    def test_execute_matching_dask_empty(self, mock_from_pandas):
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()
        mock_ddf = MagicMock()
        mock_ddf.merge.return_value = mock_ddf
        mock_ddf.shape = MagicMock()
        mock_ddf.shape.__getitem__.return_value.compute.return_value = 0
        mock_from_pandas.return_value = mock_ddf
        
        result, stats = execute_matching_dask(df1, df2, [], [])
        self.assertEqual(stats['total_match'], 0)

    @patch('dask.dataframe.from_pandas')
    def test_execute_matching_dask_with_matches(self, mock_from_pandas):
        df1 = pd.DataFrame({'key': [1, 2], 'val': ['a', 'b']})
        df2 = pd.DataFrame({'key': [2, 3], 'val': ['b', 'c']})
        
        mock_ddf = MagicMock()
        mock_ddf.merge.return_value = mock_ddf
        mock_ddf.__getitem__.return_value = mock_ddf
        mock_ddf.shape.__getitem__.return_value.compute.return_value = 1
        mock_from_pandas.return_value = mock_ddf
        
        result, stats = execute_matching_dask(df1, df2, ['key'], ['key'])
        self.assertEqual(stats['total_match'], 1)

    def test_handle_large_file_chunks(self):
        test_data = pd.DataFrame({'col': range(100)})
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = test_data
            result = handle_large_file('test.csv', chunk_size=10)
            self.assertEqual(len(result), 100)
            self.assertEqual(mock_read_csv.call_count, 1)

    def test_file_loader_process_dataframe(self):
        test_df = pd.DataFrame({
            'col1': ['1', '2', '3'],
            'col2': ['a', 'b', 'c'],
            'col3': ['\ufeff4', '5', '6']
        })
        result = FileLoader._process_dataframe(test_df)
        self.assertEqual(result['col1'].dtype, 'int64')
        self.assertEqual(list(result['col3']), ['4', '5', '6'])

    @patch('pandas.read_excel')
    def test_file_loader_excel(self, mock_read_excel):
        mock_file = MagicMock()
        mock_file.name = "test.xlsx"
        mock_file.type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        mock_file.size = 1024
        
        test_data = pd.DataFrame({'col1': [1, 2, 3]})
        mock_read_excel.return_value = test_data
        
        result = FileLoader.load_file(mock_file)
        self.assertTrue(isinstance(result, pd.DataFrame))
        mock_read_excel.assert_called_once()

if __name__ == '__main__':
    unittest.main()
