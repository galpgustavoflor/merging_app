import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
import sys
import os

# Add the path to the merging_app module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import carregar_arquivo, exibir_metadados, aplicar_regras, executar_matching

class TestApp(unittest.TestCase):

    @patch('app.st')
    def test_carregar_arquivo(self, mock_st):
        # Mock uploaded file
        mock_file = MagicMock()
        mock_file.name = 'test.csv'
        mock_file.read.return_value = b'col1,col2\n1,2\n3,4'
        
        # Test CSV file
        df = carregar_arquivo(mock_file)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape, (2, 2))
        
        # Test unsupported file format
        mock_file.name = 'test.txt'
        df = carregar_arquivo(mock_file)
        self.assertIsNone(df)
        mock_st.error.assert_called_once()

    @patch('app.st')
    @patch('app.px')
    def test_exibir_metadados(self, mock_px, mock_st):
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        exibir_metadados(df, "Test Data")
        mock_st.subheader.assert_called_once_with("Test Data")
        mock_st.dataframe.assert_called()
        mock_st.write.assert_called()
        mock_px.bar.assert_called()

    @patch('app.st')
    def test_aplicar_regras(self, mock_st):
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        mapping = {
            'col1': {'destinos': ['col2'], 'funcao': 'Direct Match'},
            'col2': {'destinos': ['col2'], 'funcao': 'Aggregation', 'transformacao': 'sum'}
        }
        mock_st.session_state.chave_origem = ['col1']
        result_df = aplicar_regras(df, mapping)
        self.assertIn('col2', result_df.columns)

    @patch('app.st')
    def test_executar_matching(self, mock_st):
        df_origem = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        df_destino = pd.DataFrame({'col1': [1, 3], 'col2': [3, 5]})
        mock_st.session_state.df_origem = df_origem
        mock_st.session_state.df_destino = df_destino
        mock_st.session_state.mapping = {
            'chave_origem': ['col1'],
            'chave_destino': ['col1'],
            'mapeamentos': {
                'col1': {'destinos': ['col1'], 'funcao': 'Direct Match'},
                'col2': {'destinos': ['col2'], 'funcao': 'Direct Match'}
            }
        }
        executar_matching()
        mock_st.write.assert_called()
        mock_st.dataframe.assert_called()

if __name__ == '__main__':
    unittest.main()