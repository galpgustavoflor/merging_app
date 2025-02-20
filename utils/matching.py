import pandas as pd
import dask.dataframe as dd
from typing import Tuple, Dict, List
import streamlit as st
from config import DASK_CONFIG

@st.cache_data
def execute_matching_dask(df_source: pd.DataFrame, df_target: pd.DataFrame, 
                         key_source: List[str], key_target: List[str]) -> Tuple[dd.DataFrame, Dict[str, int]]:
    npartitions = DASK_CONFIG["default_npartitions"]
    ddf_source = dd.from_pandas(df_source, npartitions=npartitions)
    ddf_target = dd.from_pandas(df_target, npartitions=npartitions)
    ddf_merged = ddf_source.merge(ddf_target, left_on=key_source, right_on=key_target, 
                                 how="outer", indicator=True)
    
    stats = {
        "total_match": int(ddf_merged[ddf_merged['_merge'] == 'both'].shape[0].compute()),
        "missing_source": int(ddf_merged[ddf_merged['_merge'] == 'right_only'].shape[0].compute()),
        "missing_target": int(ddf_merged[ddf_merged['_merge'] == 'left_only'].shape[0].compute())
    }
    return ddf_merged, stats
