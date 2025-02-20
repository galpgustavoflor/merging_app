import logging
import time
from functools import wraps
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
REQUESTS = Counter('app_requests_total', 'Total app requests', ['endpoint'])
PROCESSING_TIME = Histogram('app_processing_seconds', 'Time spent processing request', ['operation'])

def monitor_performance(operation):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                REQUESTS.labels(operation).inc()
                return result
            except Exception as e:
                print("Performance monitoring exception:", e)  # log exception
                raise
            finally:
                PROCESSING_TIME.labels(operation).observe(time.time() - start_time)
        return wrapper
    return decorator
