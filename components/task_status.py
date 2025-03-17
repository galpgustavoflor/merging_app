import os
import streamlit as st
import streamlit.components.v1 as components
import json
import time

# Get the directory of the current file
COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the component
def live_task_status(task_id, poll_interval=1000, height=150):
    """
    A custom component that displays live task status without requiring page reruns.
    
    Args:
        task_id: The ID of the task to monitor
        poll_interval: How often to poll for updates in milliseconds
        height: Height of the component in pixels
    """
    # Get the current task status
    from async_utils import TaskManager
    task_manager = TaskManager()
    current_status = task_manager.get_task_status(task_id)
    
    # Serialize current status to JSON for the component
    status_json = json.dumps(current_status)
    
    # Generate unique timestamp for this component instance
    timestamp = int(time.time() * 1000)
    
    # Path to the HTML file
    html_file = os.path.join(COMPONENT_DIR, "task_status.html")
    
    # Read the HTML template
    with open(html_file, "r") as f:
        html_template = f.read()
    
    # Replace placeholders with values
    html = html_template.replace("{{TASK_ID}}", task_id)
    html = html.replace("{{POLL_INTERVAL}}", str(poll_interval))
    html = html.replace("{{INITIAL_STATUS}}", status_json)
    
    # Add timestamp as HTML comment to force rerender when needed
    # This is a workaround for not having the 'key' parameter
    html = f"<!-- Component instance: {timestamp} -->\n{html}"
    
    # Display the component - without the key parameter
    components.html(html, height=height, scrolling=False)
