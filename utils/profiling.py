import pandas as pd
import dask.dataframe as dd
import streamlit as st
import plotly.express as px
from ydata_profiling import ProfileReport
from typing import Union
import numpy as np

@st.cache_data(ttl=3600)
def compute_basic_stats(_df: Union[pd.DataFrame, dd.DataFrame]) -> dict:
    """Compute basic statistics for the dataframe."""
    if isinstance(_df, dd.DataFrame):
        # Compute statistics using Dask
        stats = {
            'row_count': int(_df.shape[0].compute()),
            'column_count': _df.shape[1],
            'dtypes': _df.dtypes.to_dict(),
            'null_counts': _df.isnull().sum().compute().to_dict(),
            'memory_usage': _df.memory_usage(deep=True).sum().compute()
        }
        
        # Sample for numeric statistics
        sample = _df.sample(frac=min(10000 / int(_df.shape[0].compute()), 1)).compute()
        numeric_cols = sample.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats['numeric_summary'] = sample[numeric_cols].describe()
    else:
        # Regular pandas computations
        stats = {
            'row_count': len(_df),
            'column_count': _df.shape[1],
            'dtypes': _df.dtypes.to_dict(),
            'null_counts': _df.isnull().sum().to_dict(),
            'memory_usage': _df.memory_usage(deep=True).sum(),
            'numeric_summary': _df.describe()
        }
    
    return stats

def generate_null_value_plot(null_counts: dict, title: str):
    """Generate a bar plot for null values."""
    null_df = pd.DataFrame.from_dict(null_counts, orient='index', columns=['Null Count'])
    null_df.index.name = 'Column'
    null_df.reset_index(inplace=True)
    
    fig = px.bar(
        null_df,
        x='Column',
        y='Null Count',
        title=f'Null Values by Column - {title}',
        template='plotly_white'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        showlegend=False,
        height=400
    )
    return fig

@st.cache_resource(ttl=3600)
def generate_profile_report(df: Union[pd.DataFrame, dd.DataFrame], title: str) -> ProfileReport:
    """Generate a profile report using ydata-profiling."""
    # Convert Dask DataFrame to pandas if necessary
    if isinstance(df, dd.DataFrame):
        df = df.compute()
    elif not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be either a pandas DataFrame or a Dask DataFrame")

    profile = ProfileReport(
        df,
        title=title,
        explorative=True,
        minimal=True
    )
    return profile

def display_sample(df: Union[pd.DataFrame, dd.DataFrame], n_rows: int = 5):
    """Display a sample of the dataframe."""
    if isinstance(df, dd.DataFrame):
        sample = df.head(n_rows, compute=True)
    else:
        sample = df.head(n_rows)
    return sample
