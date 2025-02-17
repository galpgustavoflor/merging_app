import pytest
from app import execute_matching_dask

def test_matching_execution(sample_source_df, sample_target_df):
    """Test data matching execution."""
    ddf_final, stats = execute_matching_dask(
        sample_source_df,
        sample_target_df,
        ['ID'],
        ['ID_Destino']
    )
    
    assert stats['total_match'] == 5  # All source records should match
    assert stats['missing_source'] == 1  # One extra record in target
    assert stats['missing_target'] == 0  # No missing records in target

def test_matching_no_matches(sample_source_df, sample_target_df):
    """Test matching with no matching records."""
    # Modify IDs to ensure no matches
    sample_source_df['ID'] = sample_source_df['ID'] + 100
    
    ddf_final, stats = execute_matching_dask(
        sample_source_df,
        sample_target_df,
        ['ID'],
        ['ID_Destino']
    )
    
    assert stats['total_match'] == 0
    assert stats['missing_source'] == 6  # All target records unmatched
    assert stats['missing_target'] == 5  # All source records unmatched
