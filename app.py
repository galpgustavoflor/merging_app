import streamlit as st
import pandas as pd
import plotly.express as px
import json
import concurrent.futures
import dask.dataframe as dd

# The upload limit should be set in the config.toml file or via command line
# Example command line: streamlit run script.py --server.maxUploadSize=500

# Manage step states
if 'step' not in st.session_state:
    st.session_state.step = 1

if 'df_source' not in st.session_state:
    st.session_state.df_source = None

if 'df_target' not in st.session_state:
    st.session_state.df_target = None

if 'mapping' not in st.session_state:
    st.session_state.mapping = {}

if 'validation' not in st.session_state:
    st.session_state.business_rules = {}

if 'validation_rules' not in st.session_state:
    st.session_state.validation_rules = {}

if 'key_source' not in st.session_state:
    st.session_state.key_source = []

if 'key_target' not in st.session_state:
    st.session_state.key_target = []

if 'show_json' not in st.session_state:
    st.session_state.show_json = False

def load_file(uploaded_file):
    """
    Loads an Excel or CSV data file, handling formatting issues and duplicate columns.
    """
    try:
        st.write(f"Loading file: {uploaded_file.name}")
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl').convert_dtypes()
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip').convert_dtypes()
        else:
            raise ValueError("Unsupported file format. Use .xlsx or .csv")
        
        # Remove possible spaces and invisible characters from column names
        df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
        
        # Identify automatically generated duplicate columns by pandas and rename them
        renamed_columns = []
        seen_columns = {}
        new_columns = []

        for idx, col in enumerate(df.columns):
            if col in seen_columns:
                st.write(col)
                seen_columns[col] += 1
                new_name = f"{col}.{seen_columns[col]}"
                renamed_columns.append((col, new_name))
                new_columns.append(new_name)
            else:
                seen_columns[col] = 1
                new_columns.append(col)
        
        df.columns = new_columns
        
        if renamed_columns:
            st.write("### Columns renamed due to detected duplication")
            df_renamed = pd.DataFrame(renamed_columns, columns=["Original Column", "New Name"])
            st.dataframe(df_renamed)
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def display_metadata(df, title):
    """
    Displays the DataFrame metadata and interactive data quality charts.
    """
    if df is not None:
        st.subheader(title)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        st.write("Statistical Summary:")
        st.write(df.describe(include='all'))
        
        st.write("Data Types:")
        st.write(df.dtypes)
        
        st.write("Null Values by Column:")
        null_counts = df.isnull().sum().reset_index()
        null_counts.columns = ['Column', 'Null Values']
        fig_nulls = px.bar(null_counts, x='Column', y='Null Values', title='Null Values by Column')
        st.plotly_chart(fig_nulls)
    
def apply_rules(df, mapping):
    """
    Applies mapping rules, including aggregations and conversions.
    """
    st.write("Applying mapping rules...")
    
    # Filter relevant columns
    mapped_columns = [col for col, rules in mapping.items() if "destinations" in rules]
    grouping_columns = st.session_state.key_source + mapped_columns
    df = df[[col for col in grouping_columns if col in df.columns]]
    
    # Apply aggregations
    aggregations = {}
    for col, rules in mapping.items():
        if rules["function"] == "Aggregation":
            for dest_col in rules["destinations"]:
                aggregations[dest_col] = rules["transformation"]
    
    if aggregations:
        df = df.groupby(st.session_state.key_source).agg(aggregations).reset_index()
    
    # Apply conversions
    for col, rules in mapping.items():
        if rules["function"] == "Conversion":
            for dest_col in rules["destinations"]:
                try:
                    conversion_dict = json.loads(rules["transformation"])
                    df[dest_col] = df[dest_col].map(conversion_dict).fillna(df[dest_col])
                except json.JSONDecodeError:
                    st.error(f"Error converting column {dest_col}: Invalid JSON.")
    
    return df

def execute_matching_dask():
    """
    Executes the matching process using Dask to handle large volumes of data
    directly from an already loaded Pandas DataFrame.
    """
    st.write("Executing matching ...")

    df_source = st.session_state.df_source  # Already loaded as Pandas
    df_target = st.session_state.df_target  # Already loaded as Pandas

    key_source = st.session_state.key_source
    key_target = st.session_state.key_target

    # Convert df_source to a Dask DataFrame to process in parallel
    ddf_source = dd.from_pandas(df_source, npartitions=10)  # Divide into 10 partitions
    ddf_target = dd.from_pandas(df_target, npartitions=10)

    # Perform the merge using Dask
    ddf_final = ddf_source.merge(
        ddf_target, 
        left_on=key_source, 
        right_on=key_target, 
        how="outer", 
        indicator=True
    )

    # Display Matching Summary
    total_match = ddf_final[ddf_final['_merge'] == 'both'].shape[0].compute()
    missing_source = ddf_final[ddf_final['_merge'] == 'right_only'].shape[0].compute()
    missing_target = ddf_final[ddf_final['_merge'] == 'left_only'].shape[0].compute()

    st.write("### Matching Summary")
    st.write(f"Total matching records: {total_match}")
    st.write(f"Total missing in source: {missing_source}")
    st.write(f"Total missing in target: {missing_target}")

    # Convert back to Pandas to display in Streamlit (only sample to avoid memory issues)
    df_final_sample = ddf_final.compute().sample(n=min(500, len(ddf_final)), random_state=42)  # Display 500 samples

    st.write("### Matching Records")
    st.dataframe(df_final_sample[df_final_sample['_merge'] == 'both'])

    st.write("### Non-Matching Records")
    st.dataframe(df_final_sample[df_final_sample['_merge'] != 'both'])

    return ddf_final  # Keep as Dask for efficient manipulation

def execute_data_validation(df, validation_rules):
    """
    Executes data validation based on the provided validation rules.
    """
    st.write("Executing data validation...")

    validation_results = []

    for col, rules in validation_rules.items():
        total_records = len(df)
        if "validate_nulls" in rules and rules["validate_nulls"]:
            nulls = df[col].isnull().sum()
            validation_results.append({
                "Column": col,
                "Rule": "Null values",
                "Pass": total_records - nulls,
                "Fail": nulls,
                "Pass %": f"{((total_records - nulls) / total_records) * 100:.2f}%"
            })
        
        if "validate_uniqueness" in rules and rules["validate_uniqueness"]:
            unique = df[col].nunique() == len(df)
            validation_results.append({
                "Column": col,
                "Rule": "Unique values",
                "Pass": total_records if unique else 0,
                "Fail": 0 if unique else total_records - df[col].nunique(),
                "Pass %": f"{(total_records / total_records) * 100:.2f}%" if unique else "0.00%"
            })
        
        if "validate_list_of_values" in rules:
            allowed_values = rules["validate_list_of_values"]
            domain = df[~df[col].isin(allowed_values)].shape[0]
            validation_results.append({
                "Column": col,
                "Rule": "Values outside allowed list",
                "Pass": total_records - domain,
                "Fail": domain,
                "Pass %": f"{((total_records - domain) / total_records) * 100:.2f}%"
            })
        
        if "validate_regex" in rules:
            regex_pattern = rules["validate_regex"]
            regex = df[~df[col].str.match(regex_pattern, na=False)].shape[0]
            validation_results.append({
                "Column": col,
                "Rule": "Values not matching regex",
                "Pass": total_records - regex,
                "Fail": regex,
                "Pass %": f"{((total_records - regex) / total_records) * 100:.2f}%"
            })
        
        if "validate_range" in rules:
            min_value = rules.get("min_value", None)
            max_value = rules.get("max_value", None)
            if min_value is None:
                min_value = float('-inf')
            if max_value is None:
                max_value = float('inf')
            out_of_range = df[(df[col].notnull()) & ((df[col] < min_value) | (df[col] > max_value))].shape[0]
            validation_results.append({
                "Column": col,
                "Rule": "Values out of range",
                "Pass": total_records - out_of_range,
                "Fail": out_of_range,
                "Pass %": f"{((total_records - out_of_range) / total_records) * 100:.2f}%"
            })

    st.write("### Validation Results Summary")
    summary_df = pd.DataFrame(validation_results)
    st.write(summary_df.style.applymap(lambda x: 'background-color: green' if isinstance(x, str) and x.endswith('%') and float(x[:-1]) >= 90 else ('background-color: red' if isinstance(x, str) and x.endswith('%') and float(x[:-1]) < 10 else 'background-color: yellow')))

    st.write("### Detailed Validation Results")
    for col_results in validation_results:
        col = col_results["Column"]
        if col_results["Fail"] > 0:
            st.subheader(f"Column: {col}")
            if col_results["Rule"] == "Null values":
                st.write(f"Null values: {col_results['Fail']}")
                st.dataframe(df[df[col].isnull()])
            if col_results["Rule"] == "Unique values" and col_results["Fail"] > 0:
                st.write("Duplicate values found")
                st.dataframe(df[df.duplicated(subset=[col], keep=False)])
            if col_results["Rule"] == "Values outside allowed list":
                st.write(f"Values outside allowed list: {col_results['Fail']}")
                st.dataframe(df[~df[col].isin(validation_rules[col]["validate_list_of_values"])])
            if col_results["Rule"] == "Values not matching regex":
                st.write(f"Values not matching regex: {col_results['Fail']}")
                st.dataframe(df[~df[col].str.match(validation_rules[col]["validate_regex"], na=False)])
            if col_results["Rule"] == "Values out of range":
                st.write(f"Values out of range: {col_results['Fail']}")
                min_value = validation_rules[col].get("min_value", None)
                max_value = validation_rules[col].get("max_value", None)
                if min_value is None:
                    min_value = float('-inf')
                if max_value is None:
                    max_value = float('inf')
                st.dataframe(df[(df[col].notnull()) & ((df[col] < min_value) | (df[col] > max_value))])
            st.write("---")

    return validation_results

def load_validations_from_json(json_file):
    """
    Loads validation rules from a JSON file.
    """
    try:
        validation_rules = json.load(json_file)
        st.session_state.validation_rules = validation_rules
        st.success("Validation rules loaded successfully.")
        st.session_state.update_ui = True  # Flag to update the UI
    except Exception as e:
        st.error(f"Error loading validation rules: {e}")

def load_mapping_from_json(json_file):
    """
    Loads mapping rules from a JSON file.
    """
    try:
        mapping_config = json.load(json_file)
        st.session_state.mapping = mapping_config
        st.session_state.key_source = mapping_config.get("key_source", [])
        st.session_state.key_target = mapping_config.get("key_target", [])
        st.success("Mapping rules loaded successfully.")
        st.session_state.update_ui = True  # Flag to update the UI
    except Exception as e:
        st.error(f"Error loading mapping rules: {e}")

# Streamlit Interface
st.title("File Mapping Process")

if st.session_state.step == 1:
    st.header("Step 1: Load Source File")
    uploaded_source = st.file_uploader("Load Source File", type=["csv", "xlsx"])
    if uploaded_source:
        st.session_state.df_source = load_file(uploaded_source)
    if st.session_state.df_source is not None:
        display_metadata(st.session_state.df_source, "Source Data")
        if st.button("Next Step"):
            st.session_state.step = 2
            st.rerun()

if st.session_state.step == 2:
    st.header("Step 2: Load Target File")
    uploaded_target = st.file_uploader("Load Target File", type=["csv", "xlsx"])
    if uploaded_target:
        st.session_state.df_target = load_file(uploaded_target)
    if st.session_state.df_target is not None:
        display_metadata(st.session_state.df_target, "Target Data")
        if st.button("Next Step"):
            st.session_state.step = 3
            st.rerun()

if st.session_state.step == 3:
    st.header("Step 3: Define Matching Rules and Keys")
    
    with st.popover(":blue[Upload JSON File for Mapping]", icon=":material/publish:", use_container_width=True):
        uploaded_json = st.file_uploader("Load Mapping Rules from JSON", type=["json"])
        if uploaded_json:
            load_mapping_from_json(uploaded_json)
            if st.session_state.get("update_ui"):
                st.session_state.update_ui = False
#                                st.rerun()

    st.session_state.key_source = st.multiselect("Select the search key(s) in the source", st.session_state.df_source.columns, default=st.session_state.key_source)
    st.session_state.key_target = st.multiselect("Select the search key(s) in the target", st.session_state.df_target.columns, default=st.session_state.key_target)
    
    mapping_config = {"key_source": st.session_state.key_source, "key_target": st.session_state.key_target, "mappings": st.session_state.mapping.get("mappings", {})}
    
    for col in st.session_state.df_source.columns:
        if col not in st.session_state.key_source:
            with st.expander(f"Configure '{col}'"):
                option = st.radio(f"What do you want to do with '{col}'?", ["Ignore", "Map"], key=f"option_{col}", index=0 if col not in mapping_config["mappings"] else 1)
                if option == "Map":
                    mapped_cols = st.multiselect(f"Map '{col}' to:", list(st.session_state.df_target.columns), key=f"map_{col}", default=mapping_config["mappings"].get(col, {}).get("destinations", []))
                    function = st.selectbox(f"Mapping Type for '{col}'", ["Direct Match", "Aggregation", "Conversion"], key=f"func_{col}", index=["Direct Match", "Aggregation", "Conversion"].index(mapping_config["mappings"].get(col, {}).get("function", "Direct Match")))
                    transformation = mapping_config["mappings"].get(col, {}).get("transformation", None)
                    if function == "Aggregation":
                        transformation = st.selectbox("Aggregation Type", ["Sum", "Mean", "Median", "Max", "Min"], key=f"agg_{col}", index=["Sum", "Mean", "Median", "Max", "Min"].index(transformation) if transformation else 0)
                    elif function == "Conversion":
                        transformation = st.text_area("Define Conversion Dictionary (JSON)", transformation or "{}", key=f"conv_{col}")
                        try:
                            dict_data = json.loads(transformation)
                            dict_df = pd.DataFrame(list(dict_data.items()), columns=["Source", "Target"])
                            st.write("Preview of Conversion Dictionary:")
                            st.dataframe(dict_df)
                        except json.JSONDecodeError:
                            st.error("The conversion dictionary format is not valid. Make sure it is in correct JSON.")
                    mapping_config["mappings"][col] = {"destinations": mapped_cols, "function": function, "transformation": transformation}
    
    st.session_state.mapping = mapping_config
    
    if st.checkbox("Show Matching Configuration (JSON)", value=False, key="show_json"):
        st.json(st.session_state.mapping)

    if st.button("Next Step"):
        st.session_state.step = 4
        st.rerun()

if st.session_state.step == 4:
    st.header("Step 4: Execute Matching")
    if st.session_state.key_source and st.session_state.key_target:
        st.write("Matching process in progress...")
        matching_results = execute_matching_dask()
        st.session_state.matching_results = matching_results[matching_results['_merge'] == 'both']
    else:
        st.warning("Define a valid search key before executing the check.")

    if st.button("Step back"):
        st.session_state.step = 3
        st.rerun()
    
    if st.button("Configure data validations"):
        st.session_state.step = 5
        st.rerun()

if st.session_state.step == 5:
    st.header("Step 5: Mapping Validation Rules")

    business_rules_config = {}

    st.subheader("Direct Validations")
    with st.popover(":blue[Upload JSON File]", icon=":material/publish:", use_container_width=True):
        uploaded_json = st.file_uploader("Load Validation Rules from JSON", type=["json"])
        if uploaded_json:
            load_validations_from_json(uploaded_json)
            if st.session_state.get("update_ui"):
                st.session_state.update_ui = False

    validation_rules = {}
    for col in st.session_state.df_target.columns:
        with st.popover(f"Configure validation for '{col}'", use_container_width=True):
            validate_nulls = st.checkbox("Check Nulls", key=f"nulls_{col}", value=st.session_state.validation_rules.get(col, {}).get("validate_nulls", False))
            validate_unique = st.checkbox("Check Uniqueness", key=f"unique_{col}", value=st.session_state.validation_rules.get(col, {}).get("validate_uniqueness", False))
            validate_domain = st.checkbox("Check List of Values (comma separated)", key=f"domain_{col}", value=st.session_state.validation_rules.get(col, {}).get("validate_list_of_values", False))
            domain_values = st.session_state.validation_rules.get(col, {}).get("validate_list_of_values", [])
            if validate_domain:
                domain_values = st.text_input("Allowed Values", key=f"domain_values_{col}", value=",".join(domain_values))
            validate_regex = st.checkbox("Check Format (Regex)", key=f"regex_{col}", value=st.session_state.validation_rules.get(col, {}).get("validate_regex", False))
            regex_pattern = st.session_state.validation_rules.get(col, {}).get("validate_regex", "")
            if validate_regex:
                regex_pattern = st.text_input("Regular Expression", key=f"regex_pattern_{col}", value=regex_pattern)
            
            # New validation logic for numeric or date range
            if pd.api.types.is_numeric_dtype(st.session_state.df_target[col]) or pd.api.types.is_datetime64_any_dtype(st.session_state.df_target[col]):
                validate_range = st.checkbox("Check Range Value", key=f"range_{col}", value=st.session_state.validation_rules.get(col, {}).get("validate_range", False))
                if validate_range:
                    min_value = st.number_input("Minimum Value", key=f"min_{col}", value=st.session_state.validation_rules.get(col, {}).get("min_value", None))
                    max_value = st.number_input("Maximum Value", key=f"max_{col}", value=st.session_state.validation_rules.get(col, {}).get("max_value", None))
                    validation_rules[col] = {"validate_range": True, "min_value": min_value, "max_value": max_value}
            
            if validate_nulls or validate_unique or validate_domain or validate_regex:
                validation_rules[col] = validation_rules.get(col, {})
            if validate_nulls:
                validation_rules[col]["validate_nulls"] = True
            if validate_unique:
                validation_rules[col]["validate_uniqueness"] = True
            if validate_domain:
                validation_rules[col]["validate_list_of_values"] = domain_values.split(',')
            if validate_regex:
                validation_rules[col]["validate_regex"] = regex_pattern
    
    st.session_state.validation_rules = validation_rules

    if st.checkbox("Show Validation Configuration (JSON)", value=False, key="show_json"):
        st.json(st.session_state.validation_rules)

    if st.button("Next Step"):
        st.session_state.step = 6
        st.rerun()

if st.session_state.step == 6:
    st.header("Step 6: Data Validation")
    if st.session_state.df_target is not None and st.session_state.validation_rules:
        validation_results = execute_data_validation(st.session_state.df_target, st.session_state.validation_rules)
        st.session_state.validation_results = validation_results
    else:
        st.warning("Load the target data and define validation rules before executing the validation.")
    
    if st.button("Step back"):
        st.session_state.step = 5
        st.rerun()