import threading
import time
import uuid
import logging
import pandas as pd
import dask.dataframe as dd
import streamlit as st
import functools
import json
import os
from typing import Dict, Any, Callable, List, Tuple, Optional, Union
from queue import Queue
from enum import Enum

logger = logging.getLogger(__name__)

# Thread-local storage for task IDs - moved to top of file
_thread_local = threading.local()

def set_current_task_id(task_id: str) -> None:
    """Set the current task ID for this thread."""
    _thread_local.task_id = task_id

def get_current_task_id() -> Optional[str]:
    """Get the current task ID for this thread."""
    return getattr(_thread_local, 'task_id', None)

def update_task_progress(task_id: str, progress_value: float) -> None:
    """
    Update the progress of an async task.
    
    Args:
        task_id: Task ID
        progress_value: Progress value (0.0 to 1.0)
    """
    try:
        # Get task ID from thread local if not provided
        if task_id is None:
            task_id = get_current_task_id()
            if not task_id:
                logger.warning("No task ID provided and none found in thread local storage")
                return
                
        # Get task manager and update progress
        task_manager = TaskManager()
        task = task_manager._tasks.get(task_id)
        if task:
            task.update_progress(progress_value)
            logger.info(f"Updated task {task_id} progress to {progress_value:.2f}")
        else:
            logger.warning(f"Task {task_id} not found when updating progress")
    except Exception as e:
        logger.error(f"Error updating task progress: {e}")

# Create a persistent task registry to prevent "Task not found" errors
TASK_REGISTRY_FILE = os.path.join(os.path.dirname(__file__), "task_registry.json")

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class AsyncTask:
    """Represents a task that can be executed asynchronously."""
    
    def __init__(self, name: str, func: Callable, args: Tuple = None, kwargs: Dict = None):
        """
        Initialize an async task.
        
        Args:
            name: Name of the task for display purposes
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.func = func
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.status = TaskStatus.PENDING
        self.progress = 0.0
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        
    def execute(self):
        """Execute the task and capture the result or error."""
        try:
            self.status = TaskStatus.RUNNING
            self.start_time = time.time()
            self.result = self.func(*self.args, **self.kwargs)
            self.status = TaskStatus.COMPLETED
        except Exception as e:
            self.error = e
            self.status = TaskStatus.FAILED
            logger.error(f"Task {self.name} failed: {str(e)}", exc_info=True)
        finally:
            self.end_time = time.time()
            
    def update_progress(self, value: float):
        """Update task progress."""
        self.progress = max(0.0, min(1.0, value))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to a serializable dictionary for persistence."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "progress": self.progress,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": str(self.error) if self.error else None
        }

class TaskManager:
    """Manages async tasks and provides mechanisms for UI feedback."""
    
    _instance = None
    _task_queue = Queue()
    _tasks = {}
    _workers = []
    _max_workers = 3  # Maximum concurrent background tasks
    _initialized = False
    _task_registry = {}  # Persistent registry of task IDs
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TaskManager, cls).__new__(cls)
            if not cls._initialized:
                cls._initialized = True
                # Load task registry if it exists
                cls._instance._load_task_registry()
                # Call the instance method correctly
                cls._instance._start_workers()
        return cls._instance
    
    def _load_task_registry(self):
        """Load task registry from file if it exists."""
        try:
            if os.path.exists(TASK_REGISTRY_FILE):
                with open(TASK_REGISTRY_FILE, 'r') as f:
                    self._task_registry = json.load(f)
                logger.info(f"Loaded {len(self._task_registry)} tasks from registry")
        except Exception as e:
            logger.error(f"Error loading task registry: {e}")
            self._task_registry = {}
    
    def _save_task_registry(self):
        """Save task registry to file."""
        try:
            # First create a serializable version of all tasks
            serializable_tasks = {}
            for task_id, task in self._tasks.items():
                # Only save completed tasks in the registry
                if hasattr(task, 'status') and task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    serializable_tasks[task_id] = task.to_dict()
            
            # Add in existing registry entries that aren't in current tasks
            for task_id, task_info in self._task_registry.items():
                if task_id not in serializable_tasks:
                    serializable_tasks[task_id] = task_info
            
            # Save to file
            with open(TASK_REGISTRY_FILE, 'w') as f:
                json.dump(serializable_tasks, f)
            
            # Update registry
            self._task_registry = serializable_tasks
            
            logger.info(f"Saved {len(serializable_tasks)} tasks to registry")
        except Exception as e:
            logger.error(f"Error saving task registry: {e}")
    
    def _start_workers(self):
        """Start worker threads to process tasks."""
        for i in range(self._max_workers):
            # Use functools.partial to properly bind the instance method
            bound_method = functools.partial(self._worker_loop)
            worker = threading.Thread(target=bound_method, daemon=True)
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self):
        """Worker loop to process tasks from the queue."""
        while True:
            try:
                task = self._task_queue.get()
                if task is None:  # Sentinel to stop the worker
                    break
                    
                # Store task ID in thread local storage
                set_current_task_id(task.id)
                
                # Execute the task
                task.execute()
                
                # Clear task ID from thread local storage
                set_current_task_id(None)
                
                self._task_queue.task_done()
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                # Make sure to clear task ID in case of exception
                set_current_task_id(None)
    
    def task_exists(self, task_id: str) -> bool:
        """Check if a task exists in the task manager or registry."""
        if task_id is None:
            return False
        
        # Check in-memory tasks first
        if task_id in self._tasks:
            return True
            
        # Then check registry for persistent tasks
        return task_id in self._task_registry
    
    def _recover_task_from_registry(self, task_id: str) -> Optional[AsyncTask]:
        """Attempt to recover a task from the registry."""
        if task_id not in self._task_registry:
            return None
            
        # Get task info from registry
        task_info = self._task_registry.get(task_id)
        
        # Create a placeholder task
        task = AsyncTask(task_info.get("name", "Unknown Task"), lambda: None)
        task.id = task_id
        task.status = TaskStatus(task_info.get("status", TaskStatus.FAILED.value))
        task.progress = task_info.get("progress", 0.0)
        task.start_time = task_info.get("start_time")
        task.end_time = task_info.get("end_time")
        task.error = task_info.get("error")
        
        # Add to in-memory tasks and return
        self._tasks[task_id] = task
        logger.info(f"Recovered task {task_id} from registry")
        return task
    
    def submit_task(self, name: str, func: Callable, *args, **kwargs) -> str:
        """
        Submit a task for asynchronous execution.
        
        Args:
            name: Task name
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            
        Returns:
            Task ID
        """
        task = AsyncTask(name, func, args, kwargs)
        self._tasks[task.id] = task
        self._task_queue.put(task)
        logger.info(f"Task {task.id} ({name}) submitted to queue")
        return task.id
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get the status of a task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task status information
        """
        if not task_id:
            logger.warning("Attempting to get status for null task_id")
            return {
                "status": TaskStatus.FAILED.value,
                "progress": 0.0,
                "error": "Task ID is null or undefined"
            }
            
        # Try to get the task from in-memory tasks
        task = self._tasks.get(task_id)
        
        # If not found in memory, try to recover from registry
        if task is None:
            task = self._recover_task_from_registry(task_id)
        
        # If still not found, return error
        if task is None:
            logger.warning(f"Task {task_id} not found in task manager or registry")
            return {
                "status": TaskStatus.FAILED.value,
                "progress": 0.0,
                "error": f"Task not found: {task_id}"
            }
        
        result = {
            "status": task.status.value,
            "progress": task.progress,
            "result": task.result if hasattr(task, 'result') else None,
            "error": str(task.error) if task.error else None,
            "duration": task.end_time - task.start_time if task.end_time else None
        }
        
        # If task is complete, save to registry
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            self._task_registry[task_id] = task.to_dict()
            self._save_task_registry()
        
        return result
    
    def is_task_complete(self, task_id: str) -> bool:
        """Check if a task is complete."""
        # Check in-memory tasks first
        task = self._tasks.get(task_id)
        
        # If not found in memory, check registry
        if task is None and task_id in self._task_registry:
            task_info = self._task_registry.get(task_id)
            status = task_info.get("status")
            return status in (TaskStatus.COMPLETED.value, TaskStatus.FAILED.value)
            
        # If found, check status
        if task is not None:
            return task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            
        # If not found anywhere
        logger.warning(f"Checking completion status of non-existent task: {task_id}")
        return False
    
    def get_task_result(self, task_id: str, default=None):
        """Get the result of a completed task."""
        if not self.task_exists(task_id):
            logger.warning(f"Attempting to get result of non-existent task: {task_id}")
            return default
            
        task = self._tasks.get(task_id)
        if task.status != TaskStatus.COMPLETED:
            logger.warning(f"Attempting to get result of incomplete task {task_id}: {task.status}")
            return default
            
        # Add task completion tracking
        if self.is_task_complete(task_id):
            self._task_registry[task_id] = self._tasks[task_id].to_dict()
            self._save_task_registry()
            
        return self._tasks.get(task_id, AsyncTask("Missing", lambda: default)).result or default
    
    def cleanup_task(self, task_id: str):
        """Clean up a completed task to free memory."""
        if not self.task_exists(task_id):
            logger.warning(f"Attempting to clean up non-existent task: {task_id}")
            return
            
        task = self._tasks.get(task_id)
        if task.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            logger.warning(f"Attempting to clean up task {task_id} that is still {task.status}")
            return
            
        logger.info(f"Cleaning up task {task_id} ({task.name})")
        del self._tasks[task_id]
    
    def register_existing_task(self, task_id: str, name: str, status: str, result: Any = None):
        """Manually register an existing task (for recovery).
        
        Args:
            task_id: The task ID
            name: Task name
            status: Task status (one of 'pending', 'running', 'completed', 'failed')
            result: Optional task result
        """
        # Create a placeholder task
        task = AsyncTask(name, lambda: result)
        task.id = task_id
        task.status = TaskStatus(status)
        task.progress = 1.0 if status == 'completed' else 0.0
        task.result = result
        
        # Add to tasks and registry
        self._tasks[task_id] = task
        self._task_registry[task_id] = task.to_dict()
        self._save_task_registry()
        
        logger.info(f"Manually registered task {task_id} ({name}) with status {status}")
        return task_id
            
    def all_tasks_status(self) -> Dict[str, Dict]:
        """Get status of all tasks."""
        return {
            task_id: {
                "name": task.name,
                "status": task.status.value,
                "progress": task.progress
            } 
            for task_id, task in self._tasks.items()
        }

# Functions for easily submitting common data processing tasks

def async_calculate_mapping_effectiveness(matched_df: pd.DataFrame, mapping_config: Dict) -> str:
    """Submit mapping effectiveness calculation as an async task."""
    from logic import MatchingProcessor
    
    # Create a task manager instance
    task_manager = TaskManager()
    
    # Submit the task
    task_id = task_manager.submit_task(
        "Calculate Mapping Effectiveness",
        MatchingProcessor.calculate_mapping_effectiveness_sample,
        matched_df,
        mapping_config
    )
    
    return task_id

def async_execute_matching(df_source: pd.DataFrame, df_target: pd.DataFrame, 
                          mapping_config: Dict) -> str:
    """Submit matching execution as an async task."""
    from logic import MatchingProcessor
    
    # Create a task manager instance
    task_manager = TaskManager()
    
    # Submit the task
    task_id = task_manager.submit_task(
        "Execute Matching",
        MatchingProcessor.execute_matching,
        df_source,
        df_target,
        mapping_config
    )
    
    return task_id

def async_execute_validation(df: pd.DataFrame, validation_rules: Dict) -> str:
    """Submit validation execution as an async task."""
    from logic import ValidationProcessor
    
    # Create a task manager instance
    task_manager = TaskManager()
    
    # Submit the task
    task_id = task_manager.submit_task(
        "Execute Validation",
        ValidationProcessor.execute_standard_validation,
        df,
        validation_rules
    )
    
    return task_id

def async_execute_business_rules(df: pd.DataFrame, business_rules: List[Dict]) -> str:
    """Submit business rules validation as an async task."""
    from logic import ValidationProcessor
    
    # Create a task manager instance
    task_manager = TaskManager()
    
    # Submit the task
    task_id = task_manager.submit_task(
        "Execute Business Rules",
        ValidationProcessor.execute_business_rules,
        df,
        business_rules
    )
    
    return task_id

# UI helper functions to integrate with Streamlit

def display_task_progress(task_id: str, key_prefix: str = "task_progress"):
    """Display task progress in the Streamlit UI."""
    if not task_id:
        return
        
    # Create a task manager instance
    task_manager = TaskManager()
    
    # Get task status
    task_status = task_manager.get_task_status(task_id)
    status = task_status["status"]
    progress = task_status["progress"]
    
    # Display progress based on status
    if status == TaskStatus.PENDING.value:
        st.info("Task pending...")
    elif status == TaskStatus.RUNNING.value:
        progress_bar = st.progress(progress, text=f"Processing... {int(progress * 100)}%")
    elif status == TaskStatus.COMPLETED.value:
        st.success("Task completed successfully")
    elif status == TaskStatus.FAILED.value:
        st.error(f"Task failed: {task_status['error']}")

def wait_for_task_completion(task_id: str, timeout: int = 60) -> Tuple[bool, Any]:
    """
    Wait for task completion with UI feedback.
    
    Args:
        task_id: Task ID to wait for
        timeout: Maximum time to wait in seconds
        
    Returns:
        Tuple of (success, result)
    """
    if not task_id:
        return False, None
        
    # Create a task manager instance
    task_manager = TaskManager()
    
    # Create a placeholder for the progress bar
    progress_placeholder = st.empty()
    
    # Wait for completion with timeout
    start_time = time.time()
    while not task_manager.is_task_complete(task_id):
        # Check timeout
        if time.time() - start_time > timeout:
            with progress_placeholder:
                st.error(f"Task timed out after {timeout} seconds")
            return False, None
        
        # Display progress
        task_status = task_manager.get_task_status(task_id)
        with progress_placeholder:
            progress = task_status["progress"] or 0.0
            st.progress(progress, text=f"Processing... {int(progress * 100)}%")
        
        # Sleep briefly to avoid hammering the CPU
        time.sleep(0.1)
    
    # Get final status
    task_status = task_manager.get_task_status(task_id)
    
    # Display final status
    with progress_placeholder:
        if task_status["status"] == TaskStatus.COMPLETED.value:
            st.success("Task completed successfully!")
            return True, task_manager.get_task_result(task_id)
        else:
            st.error(f"Task failed: {task_status['error']}")
            return False, None

def lazy_load_component(session_key: str, loader_func: Callable, *args, **kwargs):
    """
    Lazily load a component only when it's needed.
    
    Args:
        session_key: Session state key to store the result
        loader_func: Function to load the data
        args, kwargs: Arguments for the loader function
        
    Returns:
        The loaded component
    """
    # Check if data is already in session state
    if session_key not in st.session_state:
        # Create a placeholder
        placeholder = st.empty()
        
        with placeholder:
            with st.spinner("Loading..."):
                # Load the data
                result = loader_func(*args, **kwargs)
                
                # Store in session state
                st.session_state[session_key] = result
        
        # Clear the placeholder
        placeholder.empty()
    
    return st.session_state[session_key]

def progressive_load_dataframe(df: Union[pd.DataFrame, dd.DataFrame], 
                               row_limit: int = 1000, 
                               key_prefix: str = "progressive_df"):
    """
    Progressively load a large DataFrame in chunks for better UI performance.
    
    Args:
        df: Pandas or Dask DataFrame to load
        row_limit: Maximum number of rows to load at once
        key_prefix: Session state key prefix
        
    Returns:
        The loaded DataFrame
    """
    is_dask = isinstance(df, dd.DataFrame)
    
    # Session state keys
    loaded_key = f"{key_prefix}_loaded"
    offset_key = f"{key_prefix}_offset"
    total_key = f"{key_prefix}_total"
    
    # Initialize session state
    if loaded_key not in st.session_state:
        st.session_state[loaded_key] = False
        st.session_state[offset_key] = 0
        
        # Get total rows
        if is_dask:
            with st.spinner("Computing total rows..."):
                try:
                    st.session_state[total_key] = df.shape[0].compute()
                except:
                    st.session_state[total_key] = 1000000  # Fallback estimate
        else:
            st.session_state[total_key] = len(df)
    
    # If not loaded, load a chunk
    if not st.session_state[loaded_key]:
        offset = st.session_state[offset_key]
        total = st.session_state[total_key]
        
        # Progress bar
        progress = min(1.0, offset / total) if total > 0 else 0
        st.progress(progress, text=f"Loading data {offset}/{total} rows...")
        
        # Load chunk
        end = min(offset + row_limit, total)
        try:
            if is_dask:
                chunk = df.loc[offset:end-1].compute() if offset < end else pd.DataFrame(columns=df.columns)
            else:
                chunk = df.iloc[offset:end]
                
            # Store in session state with key for this chunk
            chunk_key = f"{key_prefix}_{offset}_{end}"
            st.session_state[chunk_key] = chunk
            
            # Update offset
            st.session_state[offset_key] = end
            
            # Check if done
            if end >= total:
                st.session_state[loaded_key] = True
                st.success(f"Data loaded: {total} rows")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.session_state[loaded_key] = True  # Prevent infinite loop
    
    # Combine all loaded chunks
    chunks = []
    offset = 0
    total = st.session_state[total_key]
    
    while offset < total:
        end = min(offset + row_limit, total)
        chunk_key = f"{key_prefix}_{offset}_{end}"
        
        if chunk_key in st.session_state:
            chunks.append(st.session_state[chunk_key])
        
        offset = end
    
    # Combine chunks
    if chunks:
        return pd.concat(chunks, axis=0)
    else:
        # Return empty DataFrame with same columns
        if is_dask:
            return pd.DataFrame(columns=df.columns)
        else:
            return pd.DataFrame(columns=df.columns)

def display_async_task_status(task_id: str, auto_refresh: bool = True, 
                             key_prefix: str = "task_status", show_result: bool = False):
    """Display the status of an async task with progressive updates."""
    import time
    from async_utils import TaskManager, TaskStatus
    
    if not task_id:
        st.warning("No task ID provided")
        return None
    
    # Create task manager instance
    task_manager = TaskManager()
    
    # Check if task exists
    if not task_manager.task_exists(task_id):
        st.error(f"Task not found: {task_id}")
        return False
    
    # Get task status
    task_status = task_manager.get_task_status(task_id)
    status = task_status["status"]
    progress = task_status["progress"] or 0.0
    
    # Create status container
    status_container = st.empty()
    
    # Generate a unique key for this task display
    display_key = f"{key_prefix}_{task_id}"
    
    # Display status based on status value
    with status_container:
        if status == TaskStatus.PENDING.value:
            st.info("Task is queued and waiting to start...")
        elif status == TaskStatus.RUNNING.value:
            # Show progress bar with percentage
            st.progress(progress, text=f"Processing... {int(progress * 100)}%")
            
            # Display elapsed time if we have start_time in the task status
            if "duration" in task_status and task_status["duration"] is not None:
                st.caption(f"Running for {int(task_status['duration'])} seconds")
            
            # Auto-refresh if requested (using a random component to break caching)
            if auto_refresh:
                # We need a container to show the refresh timestamp
                refresh_container = st.empty()
                
                # Use current timestamp for refresh to avoid UI caching issues
                refresh_time = time.time()
                refresh_container.caption(f"Last update: {refresh_time}")
                
                # Rerun after a short delay
                time.sleep(0.2)
                st.rerun()
                
        elif status == TaskStatus.COMPLETED.value:
            st.success("Task completed successfully!")
            if show_result:
                result = task_manager.get_task_result(task_id)
                return result
            return True
        elif status == TaskStatus.FAILED.value:
            error_msg = task_status.get("error", "Unknown error")
            st.error(f"Task failed: {error_msg}")
            return False
    
    # Return appropriate values based on status
    if status == TaskStatus.COMPLETED.value:
        return True
    elif status == TaskStatus.FAILED.value:
        return False
    else:
        return None

def track_execution_step(key_prefix: str, step_data: Dict[str, Any], auto_refresh: bool = True) -> None:
    """
    Track and display the execution step of a multi-step process.
    
    Args:
        key_prefix: Prefix for session state keys
        step_data: Dictionary with step information (step number, total steps, message)
        auto_refresh: Whether to automatically refresh after updating step
    """
    # Generate unique keys for step tracking
    step_key = f"{key_prefix}_step"
    step_time_key = f"{key_prefix}_step_time"
    
    # Initialize step tracking in session state if not exists
    if step_key not in st.session_state:
        st.session_state[step_key] = 1
        st.session_state[step_time_key] = time.time()
    
    # Update step if new step provided
    current_step = st.session_state[step_key]
    new_step = step_data.get("step", current_step)
    
    if new_step != current_step:
        st.session_state[step_key] = new_step
        st.session_state[step_time_key] = time.time()
    
    # Extract step information
    step_num = step_data.get("step", current_step)
    total_steps = step_data.get("total_steps", 3)
    message = step_data.get("message", f"Step {step_num}/{total_steps}")
    
    # Calculate progress
    progress = min(1.0, step_num / total_steps)
    
    # Display step progress
    step_container = st.empty()
    with step_container:
        st.progress(progress, text=f"{message} ({int(progress * 100)}%)")
        
        # Show how long we've been in this step
        elapsed_time = int(time.time() - st.session_state[step_time_key])
        if elapsed_time > 5:
            st.caption(f"Running for {elapsed_time} seconds...")
    
    # Auto refresh if requested and not on final step
    if auto_refresh and step_num < total_steps:
        time.sleep(0.1)  # Brief pause
        st.rerun()
    
    return step_num
