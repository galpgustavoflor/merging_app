import streamlit as st
import pandas as pd
import plotly.express as px
import json

# O limite de upload deve ser definido no arquivo config.toml ou via linha de comando
# Exemplo de linha de comando: streamlit run script.py --server.maxUploadSize=500

# Gerenciar estado das etapas
if 'step' not in st.session_state:
    st.session_state.step = 1

if 'df_origem' not in st.session_state:
    st.session_state.df_origem = None

if 'df_destino' not in st.session_state:
    st.session_state.df_destino = None

if 'mapping' not in st.session_state:
    st.session_state.mapping = {}

if 'chave_origem' not in st.session_state:
    st.session_state.chave_origem = []

if 'chave_destino' not in st.session_state:
    st.session_state.chave_destino = []

if 'show_json' not in st.session_state:
    st.session_state.show_json = False

def carregar_arquivo(uploaded_file):
    """
    Carrega um arquivo Excel ou CSV de dados, lidando com problemas de formatação.
    """
    try:
        st.write(f"Carregando arquivo: {uploaded_file.name}")
        if uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file).convert_dtypes()
        elif uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip').convert_dtypes()
        else:
            raise ValueError("Formato de arquivo não suportado. Use .xlsx ou .csv")
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

def aplicar_regras(df, mapping):
    """
    Aplica as regras de mapeamento apenas às colunas especificadas.
    """
    st.write("Aplicando regras de mapeamento...")
    colunas_mapeadas = [regras["destinos"] for col, regras in mapping.items() if "destinos" in regras]
    colunas_mapeadas = [item for sublist in colunas_mapeadas for item in sublist]
    df = df[[col for col in colunas_mapeadas if col in df.columns]]  # Mantém apenas colunas citadas no JSON
    
    for col, regras in mapping.items():
        if "destinos" not in regras:
            continue
        for dest_col in regras["destinos"]:
            if dest_col not in df.columns:
                st.warning(f"A coluna '{dest_col}' não foi encontrada no DataFrame. Ignorando transformação.")
                continue
            
            st.write(f"Aplicando {regras['funcao']} na coluna {dest_col}")
            
            if regras["funcao"] == "Direct Match":
                continue  # Comparação direta não exige transformação
            elif regras["funcao"] == "Aggregation":
                df = df.groupby(st.session_state.chave_origem).agg({dest_col: regras["transformacao"]}).reset_index()
            elif regras["funcao"] == "Conversion":
                try:
                    conversion_dict = json.loads(regras["transformacao"])
                    df[dest_col] = df[dest_col].map(conversion_dict).fillna(df[dest_col])
                except json.JSONDecodeError:
                    st.error(f"Erro na conversão da coluna {dest_col}: JSON inválido.")
    return df

def executar_matching():
    """
    Executa a comparação entre os datasets com base nas regras de matching, utilizando apenas as colunas do JSON.
    """
    st.write("Executando matching...")
    df_origem = aplicar_regras(st.session_state.df_origem.copy(), st.session_state.mapping["mapeamentos"])
    df_destino = aplicar_regras(st.session_state.df_destino.copy(), st.session_state.mapping["mapeamentos"])
    
    # Normalizar tipos de dados antes do merge
    for col_origem, col_destino in zip(st.session_state.mapping["chave_origem"], st.session_state.mapping["chave_destino"]):
        if col_origem in df_origem.columns and col_destino in df_destino.columns:
            st.write(f"Normalizando tipos: {col_origem} e {col_destino}")
            df_origem[col_origem] = df_origem[col_origem].astype(str)
            df_destino[col_destino] = df_destino[col_destino].astype(str)
    
    st.write("Realizando merge dos datasets...")
    df_merged = df_origem.merge(df_destino, left_on=st.session_state.mapping["chave_origem"],
                                right_on=st.session_state.mapping["chave_destino"],
                                how='outer', indicator=True)
    
    st.write("### Resumo do Matching")
    st.write(f"Total de registros correspondentes: {len(df_merged[df_merged['_merge'] == 'both'])}")
    st.write(f"Total de registros faltantes na origem: {len(df_merged[df_merged['_merge'] == 'right_only'])}")
    st.write(f"Total de registros faltantes no destino: {len(df_merged[df_merged['_merge'] == 'left_only'])}")
    
    st.write("### Registros Correspondentes")
    st.dataframe(df_merged[df_merged['_merge'] == 'both'])
    
    st.write("### Registros Não Correspondentes")
    st.dataframe(df_merged[df_merged['_merge'] != 'both'])

if st.session_state.step == 4:
    st.header("Etapa 4: Execução do Matching")
    if st.session_state.chave_origem and st.session_state.chave_destino:
        st.write("Processo de Matching em execução...")
        executar_matching()
    else:
        st.warning("Defina uma chave de busca válida antes de executar a verificação.")
