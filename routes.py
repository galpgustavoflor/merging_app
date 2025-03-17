import json
import streamlit as st
from async_utils import TaskManager

def setup_task_state_handler():
    """
    Set up task state handling through session_state
    This avoids needing direct access to the server
    """
    # Initialize task status cache in session state
    if "task_status_cache" not in st.session_state:
        st.session_state.task_status_cache = {}
    
    # Register a utility function to get task status
    if "get_task_status" not in st.session_state:
        def get_task_status(task_id):
            """Get status for a specific task_id"""
            if not task_id:
                return {
                    'status': 'failed',
                    'progress': 0,
                    'error': 'No task ID provided'
                }
            
            # Get task status using TaskManager
            task_manager = TaskManager()
            return task_manager.get_task_status(task_id)
        
        st.session_state.get_task_status = get_task_status
