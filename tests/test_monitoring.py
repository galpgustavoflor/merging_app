import unittest
from unittest.mock import patch, MagicMock
import time

# Mock prometheus_client imports
class MockCounter:
    def __init__(self, *args, **kwargs):
        self.labels = MagicMock(return_value=self)
        self.inc = MagicMock()
        self._value = MagicMock()
        self._value.get.return_value = 1.0

class MockHistogram:
    def __init__(self, *args, **kwargs):
        self.labels = MagicMock(return_value=self)
        self.observe = MagicMock()
        self._sum = MagicMock()
        self._sum.get.return_value = 1.0

# Apply mocks before importing monitoring
with patch('prometheus_client.Counter', MockCounter), \
     patch('prometheus_client.Histogram', MockHistogram), \
     patch('prometheus_client.start_http_server'):
    from monitoring import monitor_performance, REQUESTS, PROCESSING_TIME

class TestMonitoring(unittest.TestCase):
    def setUp(self):
        self.counter = MockCounter()
        self.histogram = MockHistogram()
        REQUESTS.labels = self.counter.labels
        PROCESSING_TIME.labels = self.histogram.labels

    def test_monitor_performance_success(self):
        @monitor_performance("test_operation")
        def test_function():
            time.sleep(0.1)
            return "success"

        result = test_function()
        self.assertEqual(result, "success")
        self.counter.labels.assert_called_with("test_operation")
        self.counter.inc.assert_called_once()
        self.histogram.observe.assert_called_once()

    def test_monitor_performance_exception(self):
        @monitor_performance("test_error")
        def error_function():
            raise ValueError("Test error")

        with self.assertRaises(ValueError):
            error_function()
        
        self.counter.labels.assert_called_with("test_error")
        self.histogram.observe.assert_called_once()

    def test_monitor_performance_timing(self):
        start_time = time.time()
        
        @monitor_performance("test_timing")
        def timed_function():
            time.sleep(0.1)
            return "done"

        result = timed_function()
        end_time = time.time()
        
        self.assertEqual(result, "done")
        self.counter.labels.assert_called_with("test_timing")
        self.counter.inc.assert_called_once()
        
        # Verify timing observation
        call_args = self.histogram.observe.call_args[0][0]
        self.assertGreaterEqual(call_args, 0.1)
        self.assertLess(call_args, end_time - start_time + 0.1)

    def test_monitor_performance_multiple_calls(self):
        @monitor_performance("test_multiple")
        def multi_function(x):
            return x * 2

        results = [multi_function(i) for i in range(3)]
        self.assertEqual(results, [0, 2, 4])
        self.assertEqual(self.counter.inc.call_count, 3)
        self.assertEqual(self.histogram.observe.call_count, 3)

if __name__ == '__main__':
    unittest.main()
