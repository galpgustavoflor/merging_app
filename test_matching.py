import pandas as pd
import json
import streamlit as st
from utils import execute_matching_dask, apply_rules
from pathlib import Path
import numpy as np

def evaluate_mapping_results(df_source: pd.DataFrame, 
                           df_target: pd.DataFrame, 
                           df_merged: pd.DataFrame, 
                           mapping_rules: dict) -> dict:
    """Evaluate the results of the mapping process."""
    results = {
        "mapped_columns": {},
        "key_matches": {},
        "aggregation_accuracy": {},
        "conversion_accuracy": {}
    }
    
    # Get mapping configurations
    key_source = mapping_rules["key_source"]
    key_target = mapping_rules["key_target"]
    mappings = mapping_rules.get("mappings", {})
    
    # Get matched records only
    matched_records = df_merged[df_merged['_merge'] == 'both']
    
    # Apply transformations to source data first
    df_source_transformed = apply_rules(df_source, mapping_rules)
    
    # Evaluate each mapping rule
    for source_col, rule in mappings.items():
        try:
            dest_col = rule["destinations"][0]
            function = rule["function"]
            results["mapped_columns"][source_col] = dest_col
            
            # Get only matched records for comparison
            matched_source = df_source_transformed[df_source_transformed[key_source[0]].isin(matched_records[key_source[0]])]
            matched_target = df_target[df_target[key_target[0]].isin(matched_records[key_source[0]])]
            
            if function == "Aggregation":
                matches = 0
                total = len(matched_source)
                
                for idx, source_row in matched_source.iterrows():
                    target_value = matched_target[matched_target[key_target[0]] == source_row[key_source[0]]][dest_col].iloc[0]
                    if np.isclose(source_row[dest_col], target_value, rtol=1e-05):
                        matches += 1
                
                results["aggregation_accuracy"][source_col] = {
                    "matches": matches,
                    "total": total,
                    "accuracy": f"{(matches/total)*100:.2f}%" if total > 0 else "0%"
                }
                    
            elif function == "Conversion mapping":
                matches = 0
                total = len(matched_source)
                
                for idx, source_row in matched_source.iterrows():
                    target_value = matched_target[matched_target[key_target[0]] == source_row[key_source[0]]][dest_col].iloc[0]
                    if str(source_row[dest_col]).strip().upper() == str(target_value).strip().upper():
                        matches += 1
                
                results["conversion_accuracy"][source_col] = {
                    "matches": matches,
                    "total": total,
                    "accuracy": f"{(matches/total)*100:.2f}%" if total > 0 else "0%"
                }
        
        except Exception as e:
            print(f"Error processing mapping rule for {source_col}: {str(e)}")
            continue
    
    # Calculate overall key matching statistics
    results["key_matches"] = {
        "total_source_keys": df_source[key_source[0]].nunique(),
        "total_target_keys": df_target[key_target[0]].nunique(),
        "matched_keys": len(matched_records),
        "match_rate": f"{(len(matched_records)/df_source[key_source[0]].nunique())*100:.2f}%"
    }
    
    return results

def test_matching():
    # Setup paths
    base_path = Path(__file__).parent / "sample_dataset"
    
    # Load source and target data
    df_source = pd.read_csv(base_path / "exemplo_origem.csv")
    df_target = pd.read_csv(base_path / "exemplo_destino.csv")
    
    # Load mapping rules
    with open(base_path / "mapping_rules.json", 'r') as f:
        mapping_rules = json.load(f)
    
    # Set up session state
    if not hasattr(st.session_state, "mapping"):
        st.session_state.mapping = mapping_rules
    if not hasattr(st.session_state, "key_source"):
        st.session_state.key_source = mapping_rules["key_source"]
    
    try:
        print("\nExecuting matching process...")
        ddf_merged, stats = execute_matching_dask(
            df_source,
            df_target,
            mapping_rules["key_source"],
            mapping_rules["key_target"]
        )
        
        # Convert Dask DataFrame to Pandas for evaluation
        df_merged = ddf_merged.compute()
        
        # Evaluate results
        evaluation_results = evaluate_mapping_results(df_source, df_target, df_merged, mapping_rules)
        
        # Print detailed evaluation results
        print("\n=== Mapping Evaluation Results ===")
        print("\nMapped Columns:")
        for source, dest in evaluation_results["mapped_columns"].items():
            print(f"  {source} → {dest}")
        
        print("\nKey Matching Statistics:")
        for key, value in evaluation_results["key_matches"].items():
            print(f"  {key}: {value}")
        
        print("\nAggregation Accuracy:")
        for col, stats in evaluation_results["aggregation_accuracy"].items():
            print(f"  {col}:")
            print(f"    Matches: {stats['matches']}/{stats['total']} ({stats['accuracy']})")
        
        print("\nConversion Accuracy:")
        for col, stats in evaluation_results["conversion_accuracy"].items():
            print(f"  {col}:")
            print(f"    Matches: {stats['matches']}/{stats['total']} ({stats['accuracy']})")
        
        # Save evaluation results
        df_merged.to_csv(base_path / "matching_results.csv", index=False)
        with open(base_path / "evaluation_results.json", 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        print("\nResults saved to 'matching_results.csv' and 'evaluation_results.json'")
        
    except Exception as e:
        print(f"Error during matching evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    test_matching()
