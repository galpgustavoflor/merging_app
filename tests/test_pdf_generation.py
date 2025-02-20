import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
from fpdf import FPDF
from app import generate_pdf_report
from constants import Column, ValidationRule as VRule

class TestPDFGeneration(unittest.TestCase):
    def setUp(self):
        self.test_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['A', 'B', 'C']
        })
        
        self.source_summary = {
            'rows': 3,
            'columns': 2,
            'checksum': 'abc123'
        }
        
        self.target_summary = {
            'rows': 3,
            'columns': 2,
            'checksum': 'def456'
        }
        
        self.mapping = {
            'key_source': ['id'],
            'key_target': ['id'],
            'mappings': {
                'name': {
                    'destinations': ['name'],
                    'function': 'direct_mapping'
                }
            }
        }
        
        self.validation_rules = {
            'id': {
                VRule.VALIDATE_NULLS.value: True,
                VRule.VALIDATE_RANGE.value: True,
                VRule.MIN_VALUE.value: 0,
                VRule.MAX_VALUE.value: 100
            }
        }
        
        self.validation_results = [{
            Column.NAME.value: 'id',
            'Rule': 'Null values',
            'Pass': 3,
            'Fail': 0,
            'Pass %': '100.00%'
        }]

    def test_generate_pdf_report_full(self):
        """Test complete PDF report generation with all sections"""
        report_content = json.dumps({
            'test': 'data',
            'nested': {
                'key': 'value'
            }
        })
        
        pdf_data = generate_pdf_report(
            report_content,
            self.source_summary,
            self.target_summary,
            "Test checksum note",
            self.mapping,
            self.test_df,
            self.validation_rules,
            self.validation_results
        )
        
        self.assertIsInstance(pdf_data, bytes)
        self.assertTrue(len(pdf_data) > 0)

    def test_generate_pdf_report_empty_data(self):
        """Test PDF generation with minimal data"""
        report_content = json.dumps({})
        
        pdf_data = generate_pdf_report(
            report_content,
            {},
            {},
            "",
            {},
            pd.DataFrame(),
            {},
            []
        )
        
        self.assertIsInstance(pdf_data, bytes)
        self.assertTrue(len(pdf_data) > 0)

    def test_generate_pdf_report_large_data(self):
        """Test PDF generation with large dataset"""
        large_df = pd.DataFrame({
            'id': range(1000),
            'name': [f'Name_{i}' for i in range(1000)]
        })
        
        report_content = json.dumps({'large': 'dataset'})
        
        pdf_data = generate_pdf_report(
            report_content,
            self.source_summary,
            self.target_summary,
            "Test checksum note",
            self.mapping,
            large_df,
            self.validation_rules,
            self.validation_results
        )
        
        self.assertIsInstance(pdf_data, bytes)
        self.assertTrue(len(pdf_data) > 0)

    def test_generate_pdf_report_with_errors(self):
        """Test PDF generation error handling"""
        with patch('fpdf.FPDF.add_page', side_effect=Exception("Test error")):
            with self.assertRaises(Exception):
                generate_pdf_report(
                    "{}",
                    self.source_summary,
                    self.target_summary,
                    "Test checksum note",
                    self.mapping,
                    self.test_df,
                    self.validation_rules,
                    self.validation_results
                )

if __name__ == '__main__':
    unittest.main()
