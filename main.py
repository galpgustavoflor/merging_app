import streamlit as st
import logging
import time
import os
from app import main as app_main
from routes import setup_task_state_handler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

def init_app():
    """Initialize app components before the main app runs."""
    # Set up task state handler for client-server communication
    setup_task_state_handler()
    
    # Initialize any other components needed
    if "app_init_time" not in st.session_state:
        st.session_state.app_init_time = time.time()
        logger.info("App initialized")
    
    # Schedule periodic cache cleanup (every hour)
    hours_since_init = (time.time() - st.session_state.get("app_init_time", time.time())) / 3600
    if hours_since_init >= 1 or "last_cache_cleanup" not in st.session_state:
        # Clear streamlit cache
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Log memory usage before and after cleanup
        from utils import log_memory_usage
        log_memory_usage("before cache cleanup")
        import gc
        gc.collect()
        log_memory_usage("after cache cleanup")
        
        st.session_state["last_cache_cleanup"] = time.time()
        logger.info(f"Performed cache cleanup after {hours_since_init:.1f} hours")

if __name__ == "__main__":
    # Initialize app components
    init_app()
    
    # Run the main app
    app_main()
