import json
import hashlib
import datetime
import os
import uuid
import pandas as pd
import dask.dataframe as dd
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Any, Union, Optional
from fpdf import FPDF

from utils import (
    execute_matching_dask, DataValidator, generate_soda_yaml,
    validate_business_rule, validate_all_business_rules,
    clean_dataframe_for_display
)
from constants import Column, ValidationRule as VRule, Functions, COMPARISON_OPERATORS

# Set up logging
logger = logging.getLogger(__name__)

class MappingProcessor:
    """Handles mapping rule processing logic without UI dependencies"""
    
    @staticmethod
    def validate_json_structure(mapping_data: Dict) -> Tuple[bool, str]:
        """Validates the uploaded mapping structure."""
        if not isinstance(mapping_data, dict):
            return False, "Mapping must be a dictionary"
        
        if "mappings" not in mapping_data:
            return False, "Missing 'mappings' key in configuration"
            
        return True, "Valid structure"
    
    @staticmethod
    def process_mapping_config(
        mapping_config: Dict, 
        key_source: List[str], 
        key_target: List[str],
        source_columns: List[str], 
        mapped_columns: List[str],
        column_mappings: Dict[str, Dict]
    ) -> Dict:
        """Updates mapping configuration based on inputs"""
        # Start with a clean copy
        updated_config = mapping_config.copy() if mapping_config else {}
        
        # Update keys
        updated_config["key_source"] = key_source
        updated_config["key_target"] = key_target
        
        # Handle column mappings
        updated_config.setdefault("mappings", {})
        
        # Remove any columns not in mapped_columns
        for col in list(updated_config["mappings"].keys()):
            if col not in mapped_columns and col in source_columns:
                del updated_config["mappings"][col]
        
        # Update or add column mappings from column_mappings dict
        for col, config in column_mappings.items():
            if col in mapped_columns:
                updated_config["mappings"][col] = config
        
        return updated_config
    
    @staticmethod
    def parse_and_validate_transformation(transformation_text: str, function_type: str) -> Tuple[bool, str, Any]:
        """Parses and validates transformation text based on function type"""
        if function_type == Functions.CONVERSION.value:
            try:
                transformation_dict = json.loads(transformation_text or "{}")
                if not isinstance(transformation_dict, dict):
                    return False, "Transformation must be a valid dictionary", {}
                return True, "Valid transformation", transformation_dict
            except json.JSONDecodeError:
                return False, "Invalid JSON format", {}
        
        return True, "Valid transformation", transformation_text
    
    @staticmethod
    def process_mapping_ui_data(
        df_source: pd.DataFrame, 
        df_target: pd.DataFrame, 
        current_mapping: Dict,
        key_source: List[str],
        key_target: List[str],
        ui_column_mappings: Dict[str, Dict]
    ) -> Tuple[Dict, List[str]]:
        """Process mapping UI data and return updated mapping and mapped columns"""
        mapped_columns = []
        
        # Process each column mapping from UI data
        for col, config in ui_column_mappings.items():
            if config.get("destinations"):
                mapped_columns.append(col)
        
        # Update mapping configuration
        updated_config = MappingProcessor.process_mapping_config(
            current_mapping,
            key_source,
            key_target,
            df_source.columns.tolist(),
            mapped_columns,
            ui_column_mappings
        )
        
        return updated_config, mapped_columns


class MatchingProcessor:
    """Handles matching execution logic without UI dependencies"""
    
    @staticmethod
    def execute_matching(df_source: pd.DataFrame, df_target: pd.DataFrame, 
                         mapping_config: Dict) -> Tuple[dd.DataFrame, Dict[str, int], pd.DataFrame]:
        """Executes matching process and returns results"""
        try:
            # Import task progress update functionality
            # Use absolute imports instead of relative to avoid circular imports
            import sys
            import os
            
            # Add app directory to path if needed
            app_dir = os.path.dirname(os.path.abspath(__file__))
            if (app_dir not in sys.path):
                sys.path.insert(0, app_dir)
            
            # Import from local module with exception handling
            try:
                from async_utils import update_task_progress, get_current_task_id
                
                # Get the task ID for the current thread if available
                task_id = get_current_task_id()
                
                # Report initial progress
                if task_id:
                    update_task_progress(task_id, 0.05)
            except ImportError as ie:
                # Log the error but continue without task progress updates
                logger.warning(f"Task progress tracking unavailable: {str(ie)}")
                task_id = None
            
            # Log that we're using datasets (without direct st.session_state reference)
            logger.info(f"Source dataset info: {df_source.shape[0]} rows, {df_source.shape[1]} columns")
            logger.info(f"Target dataset info: {df_target.shape[0]} rows, {df_target.shape[1]} columns")
            
            # Update progress - starting data analysis
            if task_id:
                try:
                    update_task_progress(task_id, 0.1)
                except Exception as e:
                    logger.warning(f"Failed to update progress: {str(e)}")
                
            # Execute matching with mapping config
            try:
                # Update progress - starting matching
                if task_id:
                    try:
                        update_task_progress(task_id, 0.2)
                    except Exception as e:
                        logger.warning(f"Failed to update progress: {str(e)}")
                    
                ddf_merged, stats = execute_matching_dask(df_source, df_target, mapping_config)
                
                # Update progress - completed matching
                if task_id:
                    try:
                        update_task_progress(task_id, 0.6)
                    except Exception as e:
                        logger.warning(f"Failed to update progress: {str(e)}")
            except Exception as dask_error:
                # Provide more specific error messages for different types of errors
                if "Cannot serialize the return value" in str(dask_error):
                    logger.error("Caching serialization error", exc_info=True)
                    raise RuntimeError(
                        "Error with caching serialization. The cache decorator in execute_matching_dask "
                        "function should be using st.cache_resource instead of st.cache_data "
                        "since Dask DataFrames aren't serializable."
                    ) from dask_error
                elif "Can't get local object" in str(dask_error):
                    logger.error("Dask serialization error with lambda functions", exc_info=True)
                    raise RuntimeError(
                        "Error with Dask function serialization. This is likely an issue with "
                        "lambda functions inside the execute_matching_dask function."
                    ) from dask_error
                else:
                    raise
            
            # Update progress - processing results
            if task_id:
                try:
                    update_task_progress(task_id, 0.7)
                except Exception as e:
                    logger.warning(f"Failed to update progress: {str(e)}")
            
            # Ensure returned value is truly a Dask DataFrame (defensive check)
            if not isinstance(ddf_merged, dd.DataFrame):
                logger.warning("execute_matching_dask returned non-Dask DataFrame, converting to Dask")
                from config import DASK_CONFIG
                ddf_merged = dd.from_pandas(ddf_merged, npartitions=DASK_CONFIG["npartitions"])
                
            # Update progress - preparing sample
            if task_id:
                try:
                    update_task_progress(task_id, 0.8)
                except Exception as e:
                    logger.warning(f"Failed to update progress: {str(e)}")

            # Rest of the function (get sample, etc.)
            # Log that we're using datasets (without direct st.session_state reference)
            logger.info(f"Source dataset info: {df_source.shape[0]} rows, {df_source.shape[1]} columns")
            logger.info(f"Target dataset info: {df_target.shape[0]} rows, {df_target.shape[1]} columns")
            
            # Execute matching with mapping config - the dataframes should already have transformations applied
            try:
                ddf_merged, stats = execute_matching_dask(df_source, df_target, mapping_config)
            except Exception as dask_error:
                # Provide more specific error messages for different types of errors
                if "Cannot serialize the return value" in str(dask_error):
                    logger.error("Caching serialization error", exc_info=True)
                    raise RuntimeError(
                        "Error with caching serialization. The cache decorator in execute_matching_dask "
                        "function should be using st.cache_resource instead of st.cache_data "
                        "since Dask DataFrames aren't serializable."
                    ) from dask_error
                elif "Can't get local object" in str(dask_error):
                    logger.error("Dask serialization error with lambda functions", exc_info=True)
                    raise RuntimeError(
                        "Error with Dask function serialization. This is likely an issue with "
                        "lambda functions inside the execute_matching_dask function."
                    ) from dask_error
                else:
                    raise
            
            # Ensure returned value is truly a Dask DataFrame (defensive check)
            if not isinstance(ddf_merged, dd.DataFrame):
                logger.warning("execute_matching_dask returned non-Dask DataFrame, converting to Dask")
                from config import DASK_CONFIG
                ddf_merged = dd.from_pandas(ddf_merged, npartitions=DASK_CONFIG["npartitions"])
                
            # Verify we're working with a Dask DataFrame before using Dask methods
            is_dask_df = isinstance(ddf_merged, dd.DataFrame)
            logger.info(f"Merged result is Dask DataFrame: {is_dask_df}")
            
            # Calculate sample size safely based on DataFrame type
            try:
                if is_dask_df:
                    total_rows = ddf_merged.shape[0].compute()
                else:
                    total_rows = len(ddf_merged)
                sample_size = min(100, total_rows)  # Reduced from 1000 to 100
            except Exception as e:
                logger.warning(f"Error calculating sample size: {str(e)}")
                sample_size = 50  # Even smaller fallback
            
            # Verify the '_merge' column exists in the merged DataFrame
            columns = ddf_merged.columns.tolist()
            if '_merge' not in columns:
                # If missing, add a default value
                ddf_merged['_merge'] = 'unknown'
                logger.warning("'_merge' column missing in merge result, added default values")
            
            # Get a sample for display safely, considering DataFrame type
            try:
                # Create a balanced sample with both matching and non-matching records
                if is_dask_df:
                    # For Dask DataFrame, create a balanced sample with both matching and non-matching
                    try:
                        # First compute category counts to help with sampling
                        both_count = ddf_merged['_merge'].eq('both').sum().compute()
                        left_only_count = ddf_merged['_merge'].eq('left_only').sum().compute()
                        right_only_count = ddf_merged['_merge'].eq('right_only').sum().compute()
                        
                        logger.info(f"Category counts - both: {both_count}, left_only: {left_only_count}, right_only: {right_only_count}")
                        
                        # Define max records per category - dynamically adjust based on actual data distribution
                        total_sample_target = 150
                        samples = {}

                        # Improved sampling approach with better distribution
                        if both_count > 0 or left_only_count > 0 or right_only_count > 0:
                            # Calculate what percentage of total each category represents
                            total_records = both_count + left_only_count + right_only_count
                            both_percent = both_count / total_records if total_records > 0 else 0
                            left_percent = left_only_count / total_records if total_records > 0 else 0
                            right_percent = right_only_count / total_records if total_records > 0 else 0
                            
                            # Calculate sample sizes that preserve the distribution but with minimums
                            min_samples = 5  # Minimum samples to show from each category if available
                            
                            both_target = max(min_samples, int(both_percent * total_sample_target))
                            left_target = max(min_samples, int(left_percent * total_sample_target))
                            right_target = max(min_samples, int(right_percent * total_sample_target))
                            
                            logger.info(f"Sampling targets - both: {both_target}, left_only: {left_target}, right_only: {right_target}")
                            
                            # Sample "both" category
                            try:
                                if both_count > 0:
                                    # Instead of head(), use a random sampling approach for better distribution
                                    seed = np.random.randint(0, 10000)  # Random seed for reproducibility
                                    both_sample = ddf_merged[ddf_merged['_merge'].eq('both')].sample(
                                        frac=min(1.0, both_target/both_count),
                                        random_state=seed
                                    ).compute()
                                    
                                    # Limit to target size if needed
                                    if len(both_sample) > both_target:
                                        both_sample = both_sample.sample(n=both_target, random_state=seed)
                                        
                                    samples['both'] = both_sample
                                    logger.info(f"Sampled {len(both_sample)} records from 'both' category")
                            except Exception as e:
                                logger.warning(f"Error sampling 'both' category: {str(e)}")
                                
                            # Sample "left_only" category
                            try:
                                if left_only_count > 0:
                                    seed = np.random.randint(0, 10000)
                                    left_only_sample = ddf_merged[ddf_merged['_merge'].eq('left_only')].sample(
                                        frac=min(1.0, left_target/left_only_count),
                                        random_state=seed
                                    ).compute()
                                    
                                    if len(left_only_sample) > left_target:
                                        left_only_sample = left_only_sample.sample(n=left_target, random_state=seed)
                                        
                                    samples['left_only'] = left_only_sample
                                    logger.info(f"Sampled {len(left_only_sample)} records from 'left_only' category")
                            except Exception as e:
                                logger.warning(f"Error sampling 'left_only' category: {str(e)}")

                            # Sample "right_only" category
                            try:
                                if right_only_count > 0:
                                    seed = np.random.randint(0, 10000)
                                    right_only_sample = ddf_merged[ddf_merged['_merge'].eq('right_only')].sample(
                                        frac=min(1.0, right_target/right_only_count),
                                        random_state=seed
                                    ).compute()
                                    
                                    if len(right_only_sample) > right_target:
                                        right_only_sample = right_only_sample.sample(n=right_target, random_state=seed)
                                        
                                    samples['right_only'] = right_only_sample
                                    logger.info(f"Sampled {len(right_only_sample)} records from 'right_only' category")
                            except Exception as e:
                                logger.warning(f"Error sampling 'right_only' category: {str(e)}")
                                
                                # Fallback method if random sampling fails
                                try:
                                    logger.info("Attempting alternative sampling method for right_only category")
                                    # Use a different approach - take from multiple partitions
                                    partitions = min(5, ddf_merged.npartitions)
                                    parts_sample = []
                                    
                                    for i in range(partitions):
                                        # Get samples from different partitions
                                        part_df = ddf_merged.get_partition(i).compute()
                                        right_part = part_df[part_df['_merge'] == 'right_only']
                                        if len(right_part) > 0:
                                            parts_sample.append(right_part.sample(
                                                min(len(right_part), max(1, right_target // partitions))
                                            ))
                                    
                                    if parts_sample:
                                        samples['right_only'] = pd.concat(parts_sample)
                                        logger.info(f"Sampled {len(samples['right_only'])} records from 'right_only' using partition method")
                                except Exception as e2:
                                    logger.warning(f"Alternative sampling also failed: {str(e2)}")

                        # Combine samples
                        if samples:
                            # Combine samples and shuffle to ensure random distribution
                            sample_df = pd.concat(list(samples.values()), axis=0)
                            sample_df = sample_df.sample(frac=1).reset_index(drop=True)  # Shuffle the combined sample
                            
                            # Log counts in the final sample for diagnostics
                            if '_merge' in sample_df.columns:
                                sample_both = len(sample_df[sample_df['_merge'] == 'both'])
                                sample_left = len(sample_df[sample_df['_merge'] == 'left_only'])
                                sample_right = len(sample_df[sample_df['_merge'] == 'right_only'])
                                logger.info(f"Final sample: both={sample_both}, left_only={sample_left}, right_only={sample_right}")
                            
                            logger.info(f"Created balanced sample with {len(sample_df)} records")
                        else:
                            # If no samples were collected, fall back to regular sample
                            sample_df = ddf_merged.head(int(sample_size)).compute()
                            logger.info("Fell back to regular sample as no category samples were collected")
                            
                    except Exception as e1:
                        logger.warning(f"Balanced sampling approach failed: {str(e1)}")
                        # Fall back to alternative sampling methods
                        try:
                            # First approach - use direct compute
                            if sample_size > 0:
                                sample_df = ddf_merged.head(int(sample_size)).compute()
                            else:
                                sample_df = pd.DataFrame(columns=ddf_merged.columns)
                        except Exception as e1:
                            logger.warning(f"First sampling approach failed: {str(e1)}")
                            try:
                                # Second approach - convert to pandas first
                                small_ddf = ddf_merged.head(int(sample_size))
                                sample_df = small_ddf.compute() if hasattr(small_ddf, 'compute') else small_ddf
                            except Exception as e2:
                                logger.warning(f"Second sampling approach failed: {str(e2)}")
                                try:
                                    # Third approach - get a very small sample with to_pandas()
                                    sample_df = ddf_merged.head(10).to_pandas()
                                except Exception as e3:
                                    logger.error(f"All sampling approaches failed: {str(e3)}")
                                    # Last resort - empty DataFrame with right columns
                                    sample_df = pd.DataFrame(columns=ddf_merged.columns)
                else:
                    # For pandas DataFrame, create a balanced sample
                    try:
                        # First analyze distribution
                        category_counts = ddf_merged['_merge'].value_counts().to_dict()
                        logger.info(f"Category counts: {category_counts}")
                        
                        # Define sample size per category
                        max_per_category = min(50, int(sample_size/3))
                        
                        # Sample from each category
                        samples = []
                        
                        # For each category, sample proportionally but ensure minimum representation
                        for category in ['both', 'left_only', 'right_only']:
                            if category in category_counts and category_counts[category] > 0:
                                category_df = ddf_merged[ddf_merged['_merge'] == category]
                                # Take min of: 1) category size, 2) max per category
                                n_samples = min(len(category_df), max_per_category)
                                if n_samples > 0:
                                    samples.append(category_df.sample(n=n_samples))
                                    logger.info(f"Sampled {n_samples} records from '{category}' category")
                        
                        # Combine and shuffle
                        if samples:
                            sample_df = pd.concat(samples).sample(frac=1).reset_index(drop=True)
                            
                            # Log distribution in final sample
                            sample_counts = sample_df['_merge'].value_counts().to_dict()
                            logger.info(f"Final sample distribution: {sample_counts}")
                        else:
                            # Fallback
                            sample_df = ddf_merged.head(min(sample_size, len(ddf_merged)))
                    except Exception as e:
                        logger.warning(f"Pandas balanced sampling failed: {str(e)}")
                        sample_df = ddf_merged.head(min(100, len(ddf_merged)))
                
                # Double check that _merge column exists in the sample
                if '_merge' not in sample_df.columns:
                    sample_df['_merge'] = 'unknown'
                    logger.warning("Added _merge column to sample DataFrame")
                
            except Exception as e:
                logger.error(f"Unhandled error in sampling: {str(e)}", exc_info=True)
                # Last resort - create an empty DataFrame with the same columns
                try:
                    sample_df = pd.DataFrame(columns=ddf_merged.columns)
                    sample_df['_merge'] = 'unknown'
                except:
                    # Absolute last resort
                    sample_df = pd.DataFrame({'_merge': ['unknown']})
            
            # Log sample columns for debugging
            logger.info(f"Sample DataFrame columns: {sample_df.columns.tolist()}")
            
            # Calculate mapping effectiveness directly on the full matched dataset, not just the sample
            try:
                # Update progress - analyzing mapping effectiveness
                if task_id:
                    try:
                        update_task_progress(task_id, 0.85)
                    except Exception as e:
                        logger.warning(f"Failed to update progress: {str(e)}")
                
                # Extract all matched records from the full dataset, not just the sample
                if is_dask_df:
                    # For large datasets, we need to handle this differently to avoid memory issues
                    # First, check how many matched records we have
                    both_count = ddf_merged['_merge'].eq('both').sum().compute()
                    logger.info(f"Found {both_count} matched records in full dataset")
                    
                    if both_count > 0:
                        # If the matched dataset is too large, take a substantial but manageable sample
                        if both_count > 1000000:
                            # Sample size scaled based on dataset size - larger for better representation
                            sample_size = min(500000, max(50000, int(both_count * 0.1)))
                            logger.info(f"Dataset is very large ({both_count} matched records). Using {sample_size} records for mapping analysis.")
                            
                            # Extract a random sample of matched records for analysis
                            matched_records = ddf_merged[ddf_merged['_merge'].eq('both')].sample(frac=sample_size/both_count).compute()
                        else:
                            # Use all matched records if count is manageable
                            logger.info(f"Using all {both_count} matched records for mapping analysis")
                            matched_records = ddf_merged[ddf_merged['_merge'].eq('both')].compute()
                        
                        # Calculate mapping effectiveness on the matched records
                        mapping_comparison = MatchingProcessor.calculate_mapping_effectiveness_sample(
                            matched_records, mapping_config
                        )
                        stats['mapping_comparison'] = mapping_comparison
                        logger.info(f"Mapping comparison calculated with {len(mapping_comparison)} mapped columns")
                    else:
                        stats['mapping_comparison'] = {}
                        logger.info("No matched records for mapping comparison")
                else:
                    # For pandas DataFrame, use all matched records
                    matched_records = ddf_merged[ddf_merged['_merge'] == 'both']
                    if len(matched_records) > 0:
                        logger.info(f"Using all {len(matched_records)} matched records for mapping analysis")
                        mapping_comparison = MatchingProcessor.calculate_mapping_effectiveness_sample(
                            matched_records, mapping_config
                        )
                        stats['mapping_comparison'] = mapping_comparison
                        logger.info(f"Mapping comparison calculated with {len(mapping_comparison)} mapped columns")
                    else:
                        stats['mapping_comparison'] = {}
                        logger.info("No matched records for mapping comparison")
                        
            except Exception as comp_err:
                logger.error(f"Error calculating mapping comparison: {str(comp_err)}", exc_info=True)
                stats['mapping_comparison'] = {}
            
            # Update progress - finalizing
            if task_id:
                try:
                    update_task_progress(task_id, 0.95)
                except Exception as e:
                    logger.warning(f"Failed to update progress: {str(e)}")

            # Return the merged DataFrame, statistics, and sample dataframe for display
            return ddf_merged, stats, sample_df
            
        except Exception as e:
            # Re-raise with more context
            logger.error(f"Error during matching execution: {str(e)}", exc_info=True)
            raise RuntimeError(f"Error during matching execution: {str(e)}") from e

    @staticmethod
    def calculate_mapping_effectiveness_sample(matched_df: pd.DataFrame, mapping_config: Dict) -> Dict[str, Dict]:
        """
        Calculate mapping effectiveness on the entire matched dataset.
        
        Args:
            matched_df: DataFrame with only matched records (where _merge='both')
            mapping_config: Mapping configuration dictionary
            
        Returns:
            Dictionary with mapping comparison statistics for each column
        """
        if matched_df.empty:
            return {}
            
        comparison_stats = {}
        
        # Process in batches for large datasets to avoid memory issues
        batch_size = 1000000  # Process in 1M record batches for very large datasets
        total_records = len(matched_df)
        
        # Log the size of data being processed
        logger.info(f"Calculating mapping effectiveness using all {total_records} matched records")
        
        # For extremely large datasets (over 10M records), use sampling
        if total_records > 10000000:
            sample_size = 1000000  # Take 1M sample
            logger.info(f"Using {sample_size} sample for very large dataset ({total_records:,} records)")
            matched_df = matched_df.sample(sample_size)
            total_records = sample_size
        
        # Process each mapping
        for source_col, config in mapping_config.get('mappings', {}).items():
            # Get the destination column(s)
            dest_cols = config.get('destinations', [])
            if not dest_cols:
                continue
                
            # For simplicity, only compare with the first destination column
            dest_col = dest_cols[0]
            
            if source_col in matched_df.columns and dest_col in matched_df.columns:
                try:
                    # Basic statistics
                    exact_matches = (matched_df[source_col].astype(str) == matched_df[dest_col].astype(str)).sum()
                    exact_match_percentage = (exact_matches / total_records) * 100
                    
                    # Create detailed stats dictionary
                    col_stats = {
                        'total_records': total_records,
                        'exact_matches': exact_matches,
                        'exact_match_percentage': exact_match_percentage
                    }
                    
                    # Enhanced statistics: Top non-matching values
                    non_matches = matched_df[matched_df[source_col].astype(str) != matched_df[dest_col].astype(str)]
                    if not non_matches.empty:
                        # Calculate average similarity for non-matches using difflib
                        import difflib
                        from rapidfuzz import fuzz
                        
                        def calculate_similarity(row):
                            # Convert to string to handle non-string types
                            s1 = str(row[source_col]) if pd.notna(row[source_col]) else ""
                            s2 = str(row[dest_col]) if pd.notna(row[dest_col]) else ""
                            
                            # Use fuzzy matching for better similarity comparison
                            try:
                                similarity = fuzz.ratio(s1, s2)
                                return similarity
                            except Exception:
                                # Fallback to simpler comparison if rapidfuzz fails
                                try:
                                    seq = difflib.SequenceMatcher(None, s1, s2)
                                    return seq.ratio() * 100
                                except:
                                    return 0
                        
                        # Calculate similarities for a sample of non-matches (performance optimization)
                        sample_size = min(1000, len(non_matches))
                        non_match_sample = non_matches.sample(sample_size) if len(non_matches) > sample_size else non_matches
                        
                        # Calculate similarities
                        similarities = non_match_sample.apply(calculate_similarity, axis=1)
                        avg_similarity = similarities.mean()
                        col_stats['avg_similarity_non_matches'] = avg_similarity
                        
                        # Get top non-matching values (most frequent differences)
                        top_non_matches = []
                        
                        # Top values by frequency (limited to 20 for performance)
                        value_counts = non_matches[[source_col, dest_col]].value_counts().head(20)
                        for (source_value, target_value), count in value_counts.items():
                            # Calculate similarity for this specific pair
                            s1 = str(source_value) if pd.notna(source_value) else ""
                            s2 = str(target_value) if pd.notna(target_value) else ""
                            
                            try:
                                similarity = fuzz.ratio(s1, s2)
                            except:
                                similarity = 0
                                
                            top_non_matches.append({
                                'source_value': source_value,
                                'target_value': target_value,
                                'frequency': count,
                                'similarity': f"{similarity:.1f}%"
                            })
                        
                        col_stats['top_non_matches'] = top_non_matches
                        
                        # Add value distribution for the source column
                        value_dist = matched_df[source_col].value_counts().head(15).to_dict()
                        col_stats['value_distribution'] = value_dist
                    
                    # For sampling display - get a few representative examples
                    samples = []
                    # Get 5 exact matches
                    exact_match_samples = matched_df[matched_df[source_col].astype(str) == matched_df[dest_col].astype(str)].head(5)
                    for _, row in exact_match_samples.iterrows():
                        samples.append({
                            'source': row[source_col],
                            'target': row[dest_col],
                            'match': True
                        })
                    
                    # Get 5 non-matches
                    non_match_samples = matched_df[matched_df[source_col].astype(str) != matched_df[dest_col].astype(str)].head(5)
                    for _, row in non_match_samples.iterrows():
                        samples.append({
                            'source': row[source_col],
                            'target': row[dest_col],
                            'match': False
                        })
                    
                    col_stats['samples'] = samples
                    
                    # Add to the overall stats
                    comparison_stats[f"{source_col} → {dest_col}"] = col_stats
                    
                except Exception as e:
                    logger.error(f"Error calculating mapping effectiveness for {source_col} -> {dest_col}: {e}")
                    # Add basic error stats
                    comparison_stats[f"{source_col} → {dest_col}"] = {
                        'total_records': total_records,
                        'error': str(e),
                        'exact_matches': 0,
                        'exact_match_percentage': 0
                    }
        
        return comparison_stats
    
    @staticmethod
    def analyze_mapping_effectiveness(matched_df: pd.DataFrame, mapping_config: Dict) -> Dict[str, Dict]:
        """For backward compatibility - now just calls calculate_mapping_effectiveness_sample"""
        return MatchingProcessor.calculate_mapping_effectiveness_sample(matched_df, mapping_config)


class ValidationProcessor:
    """Handles data validation logic without UI dependencies"""
    
    @staticmethod
    def execute_standard_validation(df: pd.DataFrame, 
                                  validation_rules: Dict) -> List[Dict[str, Any]]:
        """Executes standard validation rules"""
        if not validation_rules:
            return []
            
        return DataValidator.execute_validation(df, validation_rules)
    
    @staticmethod
    def execute_business_rules(df: pd.DataFrame, 
                             business_rules: List[Dict]) -> Dict[str, List[int]]:
        """Executes business rule validation"""
        if not business_rules or df is None:
            return {}
            
        violations = {}
        for rule in business_rules:
            rule_violations = validate_business_rule(rule, df)
            if rule_violations:
                violations[rule['name']] = rule_violations
        
        return violations
    
    @staticmethod
    def format_rule_as_sentence(rule: dict) -> str:
        """Format a business rule as a readable sentence."""
        try:
            # Format IF conditions
            if_parts = []
            for condition in rule['conditions']:
                operator_text = {
                    'equals': 'is equal to',
                    'not_equals': 'is not equal to',
                    'greater_than': 'is greater than',
                    'less_than': 'is less than',
                    'greater_equal': 'is greater than or equal to',
                    'less_equal': 'is less than or equal to',
                    'contains': 'contains',
                    'is_null': 'is empty',
                    'is_not_null': 'is not empty'
                }.get(condition['operator'], condition['operator'])
                value_text = '' if condition['operator'] in ['is_null', 'is_not_null'] else f" {condition['value']}"
                if_parts.append(f"{condition['column']} {operator_text}{value_text}")
            if_clause = " AND ".join(if_parts)
            
            # Format THEN conditions
            then_parts = []
            for then_cond in rule['then']:
                operator_text = {
                    'equals': 'must be equal to',
                    'not_equals': 'must not be equal to',
                    'greater_than': 'must be greater than',
                    'less_than': 'must be less than',
                    'greater_equal': 'must be greater than or equal to',
                    'less_equal': 'must be less than or equal to',
                    'contains': 'must contain',
                    'is_null': 'must be empty',
                    'is_not_null': 'must not be empty'
                }.get(then_cond['operator'], then_cond['operator'])
                value_text = '' if then_cond['operator'] in ['is_null', 'is_not_null'] else f" {then_cond['value']}"
                then_parts.append(f"{then_cond['column']} {operator_text}{value_text}")
            then_clause = " AND ".join(then_parts)
            
            # Combine into full sentence
            return f"IF {if_clause}, THEN {then_clause}"
        except Exception as e:
            logger.error(f"Error formatting rule: {str(e)}")
            return "Error formatting rule"
    
    @staticmethod
    def create_business_rule(
        rule_name: str,
        conditions: List[Dict[str, Any]],
        then_conditions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create a properly structured business rule"""
        return {
            'name': rule_name,
            'conditions': conditions,
            'then': then_conditions
        }
    
    @staticmethod
    def add_rule_condition(
        column: str,
        operator: str,
        value: str,
        value_type: str = 'value'
    ) -> Dict[str, Any]:
        """Creates a properly structured rule condition"""
        return {
            'column': column,
            'operator': operator,
            'value': value,
            'value_type': value_type.lower()
        }


class ReportGenerator:
    """Generates reports without UI dependencies"""
    
    @staticmethod
    def generate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Generates data summary with checksums"""
        if df is None:
            return {}
            
        csv_data = df.to_csv(index=False).encode('utf-8')
        data_hash = hashlib.sha256(csv_data).hexdigest()
        
        return {
            "rows": df.shape[0],
            "columns": df.shape[1],
            "checksum": data_hash
        }
    
    @staticmethod
    def generate_soda_yaml_config(validation_config: Dict, 
                                table_name: str, 
                                business_rules: List = None) -> str:
        """Generates SODA YAML configuration"""
        return generate_soda_yaml(validation_config, table_name, business_rules)
    
    @staticmethod
    def generate_full_report(
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        mapping: Dict,
        matching_results: pd.DataFrame = None,
        matching_results_count: int = None,
        validation_config: Dict = None,
        validation_results: List[Dict] = None,
        business_rules: List[Dict] = None,
        business_rule_violations: Dict = None
    ) -> Dict[str, Any]:
        """Generates complete report data structure - optimized for performance."""
        # Check if we have session cache for summaries
        source_summary = ReportGenerator.generate_data_summary(source_df)
        target_summary = ReportGenerator.generate_data_summary(target_df)
        
        # Handle computing results count from either direct count or DataFrame
        if matching_results_count is not None:
            # Use provided count directly
            matching_count = matching_results_count
        elif matching_results is not None:
            # Compute from DataFrame if available
            try:
                is_dask = str(type(matching_results)) == "<class 'dask.dataframe.core.DataFrame'>"
                if is_dask:
                    if '_merge' in matching_results.columns:
                        matching_count = matching_results['_merge'].eq('both').sum().compute()
                    else:
                        matching_count = 0
                else:
                    if '_merge' in matching_results.columns:
                        matching_count = (matching_results['_merge'] == 'both').sum()
                    else:
                        matching_count = 0
            except Exception as e:
                logger.error(f"Error computing matching count: {str(e)}")
                matching_count = 0
        else:
            matching_count = 0

        # Build the report with the matching count
        report = {
            "data_audit": {
                "source": source_summary,
                "target": target_summary
            },
            "mapping": mapping,
            "validation_configuration": validation_config or {},
            "matching_results_count": int(matching_count),
            "validation_results": validation_results or [],
            "business_rules": {
                "count": len(business_rules or []),
                "violations": {name: len(violations) for name, violations in (business_rule_violations or {}).items()}
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "report_id": str(uuid.uuid4())
        }
        
        # Add report checksum
        report_json = json.dumps(report, indent=4, default=lambda o: int(o) if isinstance(o, np.integer) else str(o))
        report["checksum"] = hashlib.sha256(report_json.encode('utf-8')).hexdigest()
        
        return report
    
    @staticmethod
    def generate_pdf_report(
        report: Dict[str, Any],
        source_summary: Dict,
        target_summary: Dict,
        mapping: Dict,
        matching_results: pd.DataFrame,
        validation_config: Dict,
        validation_results: List[Dict]
    ) -> bytes:
        """Generates PDF report"""
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Corporate header
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Corporate Audit Report", ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, "Confidential", ln=True, align="C")
        pdf.ln(10)
        
        # Report content
        pdf.set_font("Arial", "", 10)
        
        # Convert report to string for display
        report_str = json.dumps(report, indent=4, default=lambda o: str(o))
        for line in report_str.splitlines():
            pdf.multi_cell(0, 8, line)
        
        # Add data audit summary page
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Data Audit Summary", ln=True, align="L")
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, "Source Dataset:", ln=True, align="L")
        pdf.cell(0, 10, f"Rows: {source_summary.get('rows', 'N/A')}, Columns: {source_summary.get('columns', 'N/A')}", ln=True, align="L")
        pdf.cell(0, 10, f"Checksum (SHA256): {source_summary.get('checksum', 'N/A')}", ln=True, align="L")
        pdf.ln(5)
        pdf.cell(0, 10, "Target Dataset:", ln=True, align="L")
        pdf.cell(0, 10, f"Rows: {target_summary.get('rows', 'N/A')}, Columns: {target_summary.get('columns', 'N/A')}", ln=True, align="L")
        pdf.cell(0, 10, f"Checksum (SHA256): {target_summary.get('checksum', 'N/A')}", ln=True, align="L")
        pdf.ln(10)
        pdf.cell(0, 10, "Checksum Note: SHA256 hash computed from the CSV representation of the dataset.", ln=True, align="L")
        
        # Add mapping configuration page
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Mapping Configuration", ln=True, align="L")
        pdf.set_font("Arial", "", 12)
        if mapping:
            pdf.cell(0, 10, f"Source Key(s): {', '.join(mapping.get('key_source', []))}", ln=True, align="L")
            pdf.cell(0, 10, f"Target Key(s): {', '.join(mapping.get('key_target', []))}", ln=True, align="L")
            pdf.ln(5)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(40, 10, "Column", 1)
            pdf.cell(60, 10, "Destinations", 1)
            pdf.cell(40, 10, "Function", 1)
            pdf.cell(50, 10, "Transformation", 1)
            pdf.ln()
            pdf.set_font("Arial", "", 12)
            for col, config in mapping.get("mappings", {}).items():
                pdf.cell(40, 10, col, 1)
                pdf.cell(60, 10, ", ".join(config.get("destinations", [])), 1)
                pdf.cell(40, 10, config.get("function", ""), 1)
                pdf.cell(50, 10, str(config.get("transformation", ""))[:20], 1)
                pdf.ln()
        else:
            pdf.cell(0, 10, "No mapping defined.", ln=True, align="L")
        
        # Add validation results page
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Validation Results Summary", ln=True, align="L")
        pdf.set_font("Arial", "", 12)
        if validation_results:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(40, 10, "Column", 1)
            pdf.cell(60, 10, "Rule", 1)
            pdf.cell(40, 10, "Pass", 1)
            pdf.cell(40, 10, "Fail", 1)
            pdf.ln()
            pdf.set_font("Arial", "", 12)
            for result in validation_results:
                pdf.cell(40, 10, result[Column.NAME.value], 1)
                pdf.cell(60, 10, result["Rule"], 1)
                pdf.cell(40, 10, str(result["Pass"]), 1)
                pdf.cell(40, 10, str(result["Fail"]), 1)
                pdf.ln()
        else:
            pdf.cell(0, 10, "No validation results available.", ln=True, align="L")
        
        # Return PDF as bytes
        return pdf.output(dest="S").encode("latin1")
    
    @staticmethod
    def generate_enhanced_pdf_report(
        report: Dict[str, Any],
        source_summary: Dict,
        target_summary: Dict,
        mapping: Dict,
        matching_results: pd.DataFrame,
        validation_config: Dict,
        validation_results: List[Dict],
        business_rules: List[Dict],
        business_rule_violations: Dict
    ) -> bytes:
        """Generates an enhanced PDF report with better formatting and visualization."""
        from fpdf import FPDF
        import matplotlib.pyplot as plt
        import io
        from PIL import Image
        import uuid
        import os
        
        class PDF(FPDF):
            def header(self):
                # Logo (replace with your company logo if available)
                #self.image('logo.png', 10, 8, 33)
                # Set font for title
                self.set_font('Arial', 'B', 15)
                # Move to the right
                self.cell(80)
                # Title
                self.cell(30, 10, 'Data Quality Report', 0, 0, 'C')
                # Draw a horizontal line
                self.line(10, 22, 200, 22)
                # Line break
                self.ln(20)
                
            def footer(self):
                # Position at 1.5 cm from bottom
                self.set_y(-15)
                # Set font
                self.set_font('Arial', 'I', 8)
                # Page number
                self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', 0, 0, 'C')
                # Date
                import datetime
                self.cell(-40, 10, datetime.datetime.now().strftime('%Y-%m-%d'), 0, 0, 'R')
                
            def chapter_title(self, title):
                self.set_font('Arial', 'B', 12)
                # Background color
                self.set_fill_color(200, 220, 255)
                # Title
                self.cell(0, 6, title, 0, 1, 'L', 1)
                # Line break
                self.ln(4)
                
            def chapter_body(self, body):
                # Set font
                self.set_font('Arial', '', 11)
                # Output text
                self.multi_cell(0, 5, body)
                # Line break
                self.ln()
                
            def add_table(self, headers, data, col_widths=None):
                # Default column widths if not specified
                if col_widths is None:
                    col_widths = [40] * len(headers)
                
                # Table header
                self.set_font('Arial', 'B', 10)
                self.set_fill_color(200, 220, 255)
                for i, header in enumerate(headers):
                    self.cell(col_widths[i], 7, header, 1, 0, 'C', 1)
                self.ln()
                
                # Table data
                self.set_font('Arial', '', 10)
                self.set_fill_color(255, 255, 255)
                for row in data:
                    for i, cell in enumerate(row):
                        self.cell(col_widths[i], 6, str(cell), 1, 0, 'L')
                    self.ln()
                
                # Space after table
                self.ln(5)
                
            # Add a custom method to safely add images from BytesIO objects
            def add_image_from_bytes(self, img_bytes, x=None, y=None, w=0, h=0, type='', link=''):
                """Safely add an image from a BytesIO object"""
                # Save to a temporary file first to avoid BytesIO issues with FPDF
                import tempfile
                
                # Create a temporary file with the proper extension
                temp = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                temp.close()
                
                try:
                    # Write bytes to the temp file
                    with open(temp.name, 'wb') as f:
                        f.write(img_bytes.getvalue())
                    
                    # Now use the file path instead of BytesIO
                    self.image(temp.name, x=x, y=y, w=w, h=h, type=type, link=link)
                finally:
                    # Clean up the temp file
                    try:
                        os.unlink(temp.name)
                    except:
                        pass
        
        # Create PDF
        pdf = PDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Executive Summary
        pdf.chapter_title('Executive Summary')
        
        # Data volume summary
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 10, 'Data Volume Summary:', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        # Source and target data summary
        source_rows = source_summary.get('rows', 'N/A')
        target_rows = target_summary.get('rows', 'N/A')
        
        pdf.cell(60, 6, 'Source Records:', 0, 0)
        pdf.cell(0, 6, f"{source_rows:,}", 0, 1)
        pdf.cell(60, 6, 'Target Records:', 0, 0)
        pdf.cell(0, 6, f"{target_rows:,}", 0, 1)
        
        # Matching summary if available
        if 'matching_results_count' in report:
            match_count = report['matching_results_count']
            match_pct = round((match_count / source_rows * 100), 2) if source_rows > 0 else 0
            
            pdf.cell(60, 6, 'Matched Records:', 0, 0)
            pdf.cell(0, 6, f"{match_count:,}", 0, 1)
            pdf.cell(60, 6, 'Match Rate:', 0, 0)
            pdf.cell(0, 6, f"{match_pct:.2f}%", 0, 1)
            
        pdf.ln(5)
        
        # Data Quality Summary
        if validation_results:
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 10, 'Data Quality Summary:', 0, 1)
            
            # Count passes and failures
            total_validations = len(validation_results)
            pass_rules = sum(1 for r in validation_results if r.get("Pass", 0) > r.get("Fail", 0))
            fail_rules = total_validations - pass_rules
            
            pdf.set_font('Arial', '', 10)
            pdf.cell(60, 6, 'Total Validation Rules:', 0, 0)
            pdf.cell(0, 6, str(total_validations), 0, 1)
            pdf.cell(60, 6, 'Passing Rules:', 0, 0)
            pdf.cell(0, 6, f"{pass_rules} ({pass_rules/total_validations*100:.1f}%)", 0, 1)
            pdf.cell(60, 6, 'Failing Rules:', 0, 0)
            pdf.cell(0, 6, f"{fail_rules} ({fail_rules/total_validations*100:.1f}%)", 0, 1)
            
            # Create a pie chart for validation results
            plt.figure(figsize=(5, 5))
            plt.pie(
                [pass_rules, fail_rules], 
                labels=['Pass', 'Fail'],
                colors=['#27AE60', '#E74C3C'],
                autopct='%1.1f%%',
                startangle=90
            )
            plt.title('Validation Rules Results')
            
            # Save plot to memory
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            buf.seek(0)
            
            # Add chart to PDF using our custom method instead of direct image()
            pdf.ln(5)
            pdf.add_image_from_bytes(buf, x=70, w=70)
            pdf.ln(5)
        
        # Business Rules Summary
        if 'business_rules' in report and report['business_rules']['count'] > 0:
            pdf.set_font('Arial', 'B', 11)
            pdf.cell(0, 10, 'Business Rules Summary:', 0, 1)
            
            rules_count = report['business_rules']['count']
            violations = report['business_rules']['violations']
            rules_with_violations = len(violations)
            
            pdf.set_font('Arial', '', 10)
            pdf.cell(60, 6, 'Total Business Rules:', 0, 0)
            pdf.cell(0, 6, str(rules_count), 0, 1)
            pdf.cell(60, 6, 'Rules with Violations:', 0, 0)
            pdf.cell(0, 6, str(rules_with_violations), 0, 1)
            
            # List rules with violations
            if violations:
                pdf.ln(5)
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 6, 'Rules with Violations:', 0, 1)
                pdf.set_font('Arial', '', 10)
                
                for rule_name, violation_count in violations.items():
                    pdf.cell(100, 6, rule_name, 0, 0)
                    pdf.cell(0, 6, f"{violation_count} violations", 0, 1)
            
            pdf.ln(5)
        
        # Detailed Data Audit
        pdf.add_page()
        pdf.chapter_title('Detailed Data Audit')
        
        # Source and target data audit
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'Source Dataset:', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(60, 6, 'Rows:', 0, 0)
        pdf.cell(0, 6, f"{source_summary.get('rows', 'N/A'):,}", 0, 1)
        pdf.cell(60, 6, 'Columns:', 0, 0)
        pdf.cell(0, 6, str(source_summary.get('columns', 'N/A')), 0, 1)
        pdf.cell(60, 6, 'Checksum (SHA256):', 0, 0)
        pdf.set_font('Courier', '', 8)  # Monospace for checksum
        pdf.cell(0, 6, source_summary.get('checksum', 'N/A')[:32] + '...', 0, 1)
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(0, 8, 'Target Dataset:', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(60, 6, 'Rows:', 0, 0)
        pdf.cell(0, 6, f"{target_summary.get('rows', 'N/A'):,}", 0, 1)
        pdf.cell(60, 6, 'Columns:', 0, 0)
        pdf.cell(0, 6, str(target_summary.get('columns', 'N/A')), 0, 1)
        pdf.cell(60, 6, 'Checksum (SHA256):', 0, 0)
        pdf.set_font('Courier', '', 8)  # Monospace for checksum
        pdf.cell(0, 6, target_summary.get('checksum', 'N/A')[:32] + '...', 0, 1)
        pdf.ln(5)
        
        # Mapping Configuration
        pdf.add_page()
        pdf.chapter_title('Mapping Configuration')
        
        if mapping:
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, 'Source Keys:', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 6, ', '.join(mapping.get('key_source', [])) or 'None')
            
            pdf.ln(2)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, 'Target Keys:', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.multi_cell(0, 6, ', '.join(mapping.get('key_target', [])) or 'None')
            
            # Mapping table
            pdf.ln(5)
            pdf.set_font('Arial', 'B', 10)
            pdf.cell(0, 6, 'Column Mappings:', 0, 1)
            
            # Create table
            headers = ['Source Column', 'Target Column(s)', 'Function']
            data = []
            
            for col, config in mapping.get('mappings', {}).items():
                destinations = ', '.join(config.get('destinations', []))
                func = config.get('function', '')
                data.append([col, destinations, func])
            
            if data:
                pdf.add_table(headers, data, col_widths=[60, 70, 60])
            else:
                pdf.set_font('Arial', 'I', 10)
                pdf.cell(0, 6, 'No mappings defined', 0, 1)
        else:
            pdf.set_font('Arial', 'I', 10)
            pdf.cell(0, 6, 'No mapping configuration available', 0, 1)
        
        # Validation Results
        if validation_results:
            pdf.add_page()
            pdf.chapter_title('Validation Results')
            
            # Group validation results by column
            from collections import defaultdict
            column_results = defaultdict(list)
            
            for result in validation_results:
                if Column.NAME.value in result:
                    column_results[result[Column.NAME.value]].append(result)
            
            # For each column, show validation results
            for column, results in column_results.items():
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 8, f'Column: {column}', 0, 1)
                
                # Create table
                headers = ['Rule', 'Pass', 'Fail', 'Pass %']
                data = []
                
                for result in results:
                    passes = result.get('Pass', 0)
                    fails = result.get('Fail', 0)
                    total = passes + fails
                    pass_pct = f"{(passes/total*100):.1f}%" if total > 0 else "0.0%"
                    
                    data.append([
                        result.get('Rule', 'Unknown'),
                        str(passes),
                        str(fails),
                        pass_pct
                    ])
                
                pdf.add_table(headers, data, col_widths=[80, 30, 30, 50])
                pdf.ln(2)
        
        # Business Rules
        if business_rules:
            pdf.add_page()
            pdf.chapter_title('Business Rules')
            
            from logic import ValidationProcessor
            
            for i, rule in enumerate(business_rules):
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 8, f"Rule {i+1}: {rule['name']}", 0, 1)
                
                pdf.set_font('Arial', '', 10)
                pdf.multi_cell(0, 6, ValidationProcessor.format_rule_as_sentence(rule))
                
                # Violations
                if rule['name'] in business_rule_violations:
                    violations = business_rule_violations[rule['name']]
                    pdf.set_fill_color(255, 200, 200)
                    pdf.cell(0, 6, f"Failed: {len(violations)} violations", 0, 1, 'L', 1)
                    
                    # Add sample violations if available
                    if len(violations) > 0:
                        pdf.cell(0, 6, "Sample violations:", 0, 1)
                        # Create a small table with up to 5 sample violations
                        sample_size = min(5, len(violations))
                        if sample_size > 0 and isinstance(matching_results, pd.DataFrame):
                            try:
                                sample_indices = violations[:sample_size]
                                sample_data = []
                                for idx in sample_indices:
                                    try:
                                        row_data = []
                                        for col in matching_results.columns[:4]:  # First 4 columns only
                                            val = str(matching_results.iloc[idx][col])
                                            # Truncate long values
                                            if len(val) > 15:
                                                val = val[:12] + "..."
                                            row_data.append(val)
                                        sample_data.append(row_data)
                                    except IndexError:
                                        pass
                                
                                if sample_data:
                                    headers = matching_results.columns[:4].tolist()
                                    pdf.add_table(headers, sample_data, col_widths=[45, 45, 45, 45])
                            except Exception as e:
                                pdf.cell(0, 6, f"Error displaying samples: {str(e)}", 0, 1)
                else:
                    pdf.set_fill_color(200, 255, 200)
                    pdf.cell(0, 6, "Passed: No violations", 0, 1, 'L', 1)
                
                pdf.ln(5)
        
        # Add report metadata
        pdf.add_page()
        pdf.chapter_title('Report Metadata')
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(60, 6, 'Report Generated:', 0, 0)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, report.get('timestamp', 'N/A'), 0, 1)
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(60, 6, 'Report Checksum:', 0, 0)
        pdf.set_font('Courier', '', 8)
        pdf.cell(0, 6, report.get('checksum', 'N/A'), 0, 1)
        
        # Add audit information
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 6, 'Audit Information:', 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(60, 6, 'Report ID:', 0, 0)
        pdf.cell(0, 6, report.get('report_id', str(uuid.uuid4())), 0, 1)
        
        pdf.cell(60, 6, 'User:', 0, 0)
        pdf.cell(0, 6, os.environ.get('USERNAME', 'Unknown'), 0, 1)
        
        # Add execution environment information
        pdf.cell(60, 6, 'Report Date:', 0, 0)
        pdf.cell(0, 6, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 0, 1)
        
        # Add data flow diagram if available
        try:
            if mapping and mapping.get('mappings'):
                pdf.add_page()
                pdf.chapter_title('Data Flow Diagram')
                
                # Create a diagram showing data flow from source to target
                plt.figure(figsize=(8, 4))
                
                # Get source and target columns from mapping
                source_columns = list(mapping.get('mappings', {}).keys())
                
                # Get all destination columns
                target_columns = []
                for config in mapping.get('mappings', {}).values():
                    target_columns.extend(config.get('destinations', []))
                target_columns = list(set(target_columns))
                
                # Create a simple visualization if libraries are available
                try:
                    import networkx as nx
                    
                    # Create a directed graph
                    G = nx.DiGraph()
                    
                    # Add source and target nodes
                    for col in source_columns:
                        G.add_node(f"S:{col}", bipartite=0)
                    
                    for col in target_columns:
                        G.add_node(f"T:{col}", bipartite=1)
                    
                    # Add edges based on mapping
                    for src_col, config in mapping.get('mappings', {}).items():
                        for dest_col in config.get('destinations', []):
                            G.add_edge(f"S:{src_col}", f"T:{dest_col}")
                    
                    # Position nodes
                    pos = {}
                    # Source nodes on the left
                    src_nodes = [node for node in G.nodes() if node.startswith("S:")]
                    for i, node in enumerate(src_nodes):
                        pos[node] = (0, (i - len(src_nodes) / 2) * 0.8)
                    
                    # Target nodes on the right
                    tgt_nodes = [node for node in G.nodes() if node.startswith("T:")]
                    for i, node in enumerate(tgt_nodes):
                        pos[node] = (1, (i - len(tgt_nodes) / 2) * 0.8)
                    
                    # Draw the graph
                    plt.figure(figsize=(10, max(6, len(source_columns) * 0.4)))
                    nx.draw_networkx_nodes(G, pos, nodelist=src_nodes, node_color='#FF6600', node_size=500, alpha=0.8)
                    nx.draw_networkx_nodes(G, pos, nodelist=tgt_nodes, node_color='#3498DB', node_size=500, alpha=0.8)
                    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrowsize=20)
                    
                    # Add labels with better visibility
                    labels = {node: node.split(":", 1)[1] for node in G.nodes()}
                    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')
                    
                    plt.title("Data Mapping Flow")
                    plt.axis('off')
                    
                    # Save to memory
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    plt.close()
                    buf.seek(0)
                    
                    # Add to PDF using our custom method
                    pdf.add_image_from_bytes(buf, x=10, w=190)
                
                except ImportError:
                    # Fall back to a simpler representation if networkx not available
                    pdf.multi_cell(0, 6, "Data flow visualization requires the networkx library.")
                    
                    # Create a simple table instead
                    headers = ['Source Column', 'Target Column']
                    data = []
                    for src_col, config in mapping.get('mappings', {}).items():
                        for dest_col in config.get('destinations', []):
                            data.append([src_col, dest_col])
                    
                    if data:
                        pdf.add_table(headers, data, col_widths=[95, 95])
        
        except Exception as e:
            logger.error(f"Failed to generate data flow diagram: {str(e)}")
            pdf.cell(0, 10, f"Error generating data flow diagram: {str(e)}", 0, 1)
        
        # Add disclaimers and notes
        pdf.add_page()
        pdf.chapter_title('Notes and Disclaimers')
        
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 10, 'Notes:', 0, 1)
        
        pdf.set_font('Arial', '', 9)
        pdf.multi_cell(0, 5, "This report was generated automatically by the Dataset Comparison and Validation Tool. "
                       "The analysis is based on the data provided at the time of generation and may not reflect "
                       "subsequent changes to the source systems.")
        
        pdf.ln(5)
        pdf.multi_cell(0, 5, "Data validation rules and business rules were applied as configured in the application. "
                       "For the most accurate interpretation of results, please consult with your data team.")
        
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 10)
        pdf.cell(0, 10, 'Disclaimer:', 0, 1)
        
        pdf.set_font('Arial', 'I', 9)
        pdf.multi_cell(0, 5, "This document is for informational purposes only. While every effort has been made to ensure "
                       "the accuracy and completeness of the information, no guarantee is given nor responsibility taken "
                       "for errors or omissions in this document.")
        
        # Return PDF as bytes
        return pdf.output(dest="S").encode("latin1")
    
    @staticmethod
    def generate_comprehensive_report(
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        matching_results: pd.DataFrame,
        mapping_config: Dict,
        matching_stats: Dict,
        validation_results: List[Dict],
        business_rules: List[Dict],
        business_rule_violations: Dict
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive report with all key information consolidated.
        
        Returns:
            Dict containing all report components 
        """
        report_data = {}
        
        # Generate source and target summaries
        report_data["source_summary"] = ReportGenerator.generate_data_summary(source_df)
        report_data["target_summary"] = ReportGenerator.generate_data_summary(target_df)
        
        # Calculate key metrics
        total_match = int(matching_stats.get('total_match', 0))
        missing_source = int(matching_stats.get('missing_source', 0))
        missing_target = int(matching_stats.get('missing_target', 0))
        total_records = total_match + missing_source + missing_target
        
        # Prepare matching metrics
        report_data["matching_metrics"] = {
            "total_match": total_match,
            "missing_source": missing_source,
            "missing_target": missing_target,
            "total_records": total_records,
            "match_percentage": round((total_match / total_records * 100), 2) if total_records > 0 else 0
        }
        
        # Prepare validation metrics
        total_validations = len(validation_results)
        pass_validations = sum(1 for r in validation_results if r.get("Pass", 0) > r.get("Fail", 0))
        
        report_data["validation_metrics"] = {
            "total_rules": total_validations,
            "pass_rules": pass_validations,
            "fail_rules": total_validations - pass_validations,
            "pass_percentage": round((pass_validations / total_validations * 100), 2) if total_validations > 0 else 0
        }
        
        # Prepare business rules metrics
        total_rules = len(business_rules)
        rules_with_violations = len(business_rule_violations)
        total_violations = sum(len(violations) for violations in business_rule_violations.values())
        
        report_data["business_rules_metrics"] = {
            "total_rules": total_rules,
            "rules_with_violations": rules_with_violations,
            "rules_passing": total_rules - rules_with_violations,
            "total_violations": total_violations,
            "pass_percentage": round(((total_rules - rules_with_violations) / total_rules * 100), 2) if total_rules > 0 else 0
        }
        
        # Extract unmatched records for CSV export
        if matching_results is not None and '_merge' in matching_results.columns:
            # Only take up to 10,000 records for each to avoid memory issues
            try:
                missing_in_source = get_dataframe_sample(matching_results[matching_results['_merge'] == 'right_only'], 10000)
                missing_in_target = get_dataframe_sample(matching_results[matching_results['_merge'] == 'left_only'], 10000)
                
                # Remove the _merge column for cleaner exports
                if '_merge' in missing_in_source.columns:
                    missing_in_source = missing_in_source.drop(columns=['_merge'])
                if '_merge' in missing_in_target.columns:
                    missing_in_target = missing_in_target.drop(columns=['_merge'])
                    
                report_data["unmatched_exports"] = {
                    "missing_in_source": missing_in_source,
                    "missing_in_target": missing_in_target
                }
            except Exception as e:
                logger.error(f"Error preparing unmatched exports: {e}")
                report_data["unmatched_exports"] = None
        else:
            report_data["unmatched_exports"] = None
            
        # Store validation configuration for SODA export
        report_data["validation_config"] = {}
        
        # Extract validation rules from validation results
        for result in validation_results:
            col_name = result.get(Column.NAME.value)
            rule_type = result.get("Rule")
            
            if col_name and rule_type:
                # Initialize column in validation config if not present
                if col_name not in report_data["validation_config"]:
                    report_data["validation_config"][col_name] = {}
                
                # Map rule types to validation config
                if rule_type == "Null values":
                    report_data["validation_config"][col_name][VRule.VALIDATE_NULLS.value] = True
                elif rule_type == "Unique values":
                    report_data["validation_config"][col_name][VRule.VALIDATE_UNIQUENESS.value] = True
                elif rule_type == "Values outside allowed list":
                    # Need to get allowed values from original validation rules
                    # Just placeholder for now
                    report_data["validation_config"][col_name][VRule.VALIDATE_LIST_OF_VALUES.value] = []
                elif rule_type == "Values not matching regex":
                    # Need to get regex pattern from original validation rules
                    # Just placeholder for now
                    report_data["validation_config"][col_name][VRule.VALIDATE_REGEX.value] = ".*"
                elif rule_type == "Values out of range":
                    report_data["validation_config"][col_name][VRule.VALIDATE_RANGE.value] = True
        
        # Generate SODA validation config
        report_data["soda_config"] = generate_soda_yaml(
            report_data["validation_config"], 
            "data_quality_validation",  # default table name
            business_rules
        )
        
        # Add premises summary (mapping and validation configurations)
        report_data["premises"] = {
            "mapping_config": mapping_config,
            "key_columns": {
                "source": mapping_config.get('key_source', []),
                "target": mapping_config.get('key_target', [])
            },
            "mapped_columns_count": len(mapping_config.get('mappings', {})),
            "validation_rules_count": total_validations,
            "business_rules_count": total_rules
        }
        
        # Add timestamp
        import datetime
        report_data["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate report ID
        import hashlib
        report_checksum = hashlib.md5(
            f"{report_data['timestamp']}_{total_match}_{total_validations}".encode()
        ).hexdigest()[:8]
        report_data["checksum"] = report_checksum
        
        return report_data


# Additional standalone functions that don't fit into any of the classes above

def convert_pandas_to_dask_if_needed(df):
    """Convert pandas DataFrame to Dask DataFrame if needed"""
    if df is not None and isinstance(df, pd.DataFrame) and not isinstance(df, dd.DataFrame):
        from config import DASK_CONFIG
        return dd.from_pandas(df, npartitions=DASK_CONFIG["npartitions"])
    return df

def ensure_matching_results_are_dask(matching_results):
    """Ensure matching_results is a Dask DataFrame"""
    if matching_results is not None:
        try:
            # First check explicitly if it's already a Dask DataFrame
            if str(type(matching_results)) == "<class 'dask.dataframe.core.DataFrame'>":
                logger.info("Matching results already a Dask DataFrame")
                return matching_results
                
            # If not, check if it's a pandas DataFrame
            if isinstance(matching_results, pd.DataFrame):
                logger.info("Converting pandas DataFrame to Dask DataFrame")
                try:
                    from config import DASK_CONFIG
                    matching_results = dd.from_pandas(matching_results, npartitions=DASK_CONFIG["npartitions"])
                    logger.info("Successfully converted matching results to Dask DataFrame")
                    return matching_results
                except Exception as e:
                    logger.error(f"Error converting to Dask DataFrame: {str(e)}", exc_info=True)
                    logger.warning("Continuing with pandas DataFrame")
            
            # If it's neither, log a warning
            if not isinstance(matching_results, pd.DataFrame):
                logger.warning(f"Unexpected type for matching_results: {type(matching_results)}")
        except Exception as e:
            logger.error(f"Error checking DataFrame type: {str(e)}", exc_info=True)
    
    return matching_results

def get_dataframe_sample(df, sample_size=100):
    """Get a sample from a DataFrame, handling both pandas and Dask DataFrames"""
    if df is None:
        return pd.DataFrame()
    try:
        # Check if df is a Dask DataFrame by looking for dask attributes
        if hasattr(df, 'compute') and hasattr(df, 'npartitions'):
            sample_df = df.head(sample_size)
            # Only call compute if the returned sample has that method
            if hasattr(sample_df, 'compute'):
                return sample_df.compute()
            else:
                return sample_df
        else:
            actual_sample_size = min(sample_size, len(df))
            return df.head(actual_sample_size)
    except Exception as e:
        logger.error(f"Error getting DataFrame sample: {str(e)}")
        if hasattr(df, 'columns'):
            return pd.DataFrame(columns=df.columns)
        else:
            return pd.DataFrame()
