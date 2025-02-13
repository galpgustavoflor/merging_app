import streamlit as st
import pandas as pd
import plotly.express as px
import json
import concurrent.futures

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
    Carrega um arquivo Excel ou CSV de dados, lidando com problemas de formatação e colunas duplicadas.
    """
    try:
        st.write(f"Carregando arquivo: {uploaded_file.name}")
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl').convert_dtypes()
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, sep=None, engine='python', on_bad_lines='skip').convert_dtypes()
        else:
            raise ValueError("Formato de arquivo não suportado. Use .xlsx ou .csv")
        
        # Remover possíveis espaços e caracteres invisíveis nos nomes das colunas
        df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
        
        # Identificar colunas duplicadas geradas automaticamente pelo pandas e renomeá-las
        colunas_renomeadas = []
        colunas_vistas = {}
        novas_colunas = []
        #st.write(df.columns)
        for idx, col in enumerate(df.columns):
            if col in colunas_vistas:
                st.write(col)
                colunas_vistas[col] += 1
                novo_nome = f"{col}.{colunas_vistas[col]}"
                colunas_renomeadas.append((col, novo_nome))
                novas_colunas.append(novo_nome)
            else:
                colunas_vistas[col] = 1
                novas_colunas.append(col)
        
        df.columns = novas_colunas
        
        if colunas_renomeadas:
            st.write("### Colunas renomeadas devido a duplicação detectada")
            df_renomeadas = pd.DataFrame(colunas_renomeadas, columns=["Coluna Original", "Novo Nome"])
            st.dataframe(df_renomeadas)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return None

def exibir_metadados(df, titulo):
    """
    Exibe os metadados do DataFrame e gráficos interativos de qualidade de dados.
    """
    if df is not None:
        st.subheader(titulo)
        st.write("Visualização dos dados:")
        st.dataframe(df.head())
        
        st.write("Resumo Estatístico:")
        st.write(df.describe(include='all'))
        
        st.write("Tipos de Dados:")
        st.write(df.dtypes)
        
        st.write("Valores Nulos por Coluna:")
        null_counts = df.isnull().sum().reset_index()
        null_counts.columns = ['Coluna', 'Valores Nulos']
        fig_nulls = px.bar(null_counts, x='Coluna', y='Valores Nulos', title='Valores Nulos por Coluna')
        st.plotly_chart(fig_nulls)
    
def aplicar_regras(df, mapping):
    """
    Aplica as regras de mapeamento, incluindo agregações e conversões.
    """
    st.write("Aplicando regras de mapeamento...")
    st.write(mapping)
    
    # Filtrar colunas relevantes
    colunas_mapeadas = [col for col, regras in mapping.items() if "destinos" in regras]
    colunas_agrupamento = st.session_state.chave_origem + colunas_mapeadas
    df = df[[col for col in colunas_agrupamento if col in df.columns]]
    
    # Aplicar agregações
    agregacoes = {}
    for col, regras in mapping.items():
        if regras["funcao"] == "Aggregation":
            for dest_col in regras["destinos"]:
                agregacoes[dest_col] = regras["transformacao"]
    
    if agregacoes:
        df = df.groupby(st.session_state.chave_origem).agg(agregacoes).reset_index()
    
    # Aplicar conversões
    for col, regras in mapping.items():
        if regras["funcao"] == "Conversion":
            for dest_col in regras["destinos"]:
                try:
                    conversion_dict = json.loads(regras["transformacao"])
                    df[dest_col] = df[dest_col].map(conversion_dict).fillna(df[dest_col])
                except json.JSONDecodeError:
                    st.error(f"Erro na conversão da coluna {dest_col}: JSON inválido.")
    
    return df

def process_chunk(df_origem_chunk, df_destino, chave_origem, chave_destino):
    df_merged = df_origem_chunk.merge(
        df_destino,
        left_on=chave_origem,
        right_on=chave_destino,
        how='outer',
        indicator=True
    )
    return df_merged

def executar_matching():
    """
    Executa o processo de matching usando processamento em lotes para reduzir o consumo de memória.
    """
    st.write("Executando matching em lotes para otimização de memória...")
    chunk_size = 10000  # Define o tamanho do lote para evitar sobrecarga de memória
    
    df_origem = st.session_state.df_origem.copy()
    df_destino = st.session_state.df_destino.copy()
    
    # Converter colunas numéricas para tipos menores
    for col in df_origem.select_dtypes(include=['int64', 'float64']).columns:
        df_origem[col] = pd.to_numeric(df_origem[col], downcast='integer')
    for col in df_destino.select_dtypes(include=['int64', 'float64']).columns:
        df_destino[col] = pd.to_numeric(df_destino[col], downcast='integer')
    
    # Normalizar chaves
    for col_origem, col_destino in zip(st.session_state.mapping["chave_origem"], st.session_state.mapping["chave_destino"]):
        if col_origem in df_origem.columns and col_destino in df_destino.columns:
            df_origem[col_origem] = df_origem[col_origem].astype(str)
            df_destino[col_destino] = df_destino[col_destino].astype(str)
    
    # Processamento em lotes com paralelismo
    resultado = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(df_origem), chunk_size):
            df_origem_chunk = df_origem.iloc[i:i+chunk_size]
            st.write(f"{i+chunk_size} linhas processadas...")
            chave_origem = st.session_state.mapping["chave_origem"]
            chave_destino = st.session_state.mapping["chave_destino"]
            futures.append(executor.submit(process_chunk, df_origem_chunk, df_destino, chave_origem, chave_destino))
        
        for future in concurrent.futures.as_completed(futures):
            resultado.append(future.result())
    
    df_final = pd.concat(resultado, ignore_index=True)
    
    st.write("### Resumo do Matching")
    st.write(f"Total de registros correspondentes: {len(df_final[df_final['_merge'] == 'both'])}")
    st.write(f"Total de registros faltantes na origem: {len(df_final[df_final['_merge'] == 'right_only'])}")
    st.write(f"Total de registros faltantes no destino: {len(df_final[df_final['_merge'] == 'left_only'])}")
    
    st.write("### Registros Correspondentes")
    st.dataframe(df_final[df_final['_merge'] == 'both'])
    
    st.write("### Registros Não Correspondentes")
    st.dataframe(df_final[df_final['_merge'] != 'both'])

# Interface Streamlit
st.title("Processo de Mapeamento de Arquivos")

if st.session_state.step == 1:
    st.header("Etapa 1: Carregar Arquivo de Origem")
    uploaded_origem = st.file_uploader("Carregar Arquivo de Origem", type=["csv", "xlsx"])
    if uploaded_origem:
        st.session_state.df_origem = carregar_arquivo(uploaded_origem)
    if st.session_state.df_origem is not None:
        exibir_metadados(st.session_state.df_origem, "Dados de Origem")
        if st.button("Próxima Etapa"):
            st.session_state.step = 2
            st.rerun()

if st.session_state.step == 2:
    st.header("Etapa 2: Carregar Arquivo de Destino")
    uploaded_destino = st.file_uploader("Carregar Arquivo de Destino", type=["csv", "xlsx"])
    if uploaded_destino:
        st.session_state.df_destino = carregar_arquivo(uploaded_destino)
    if st.session_state.df_destino is not None:
        exibir_metadados(st.session_state.df_destino, "Dados de Destino")
        if st.button("Próxima Etapa"):
            st.session_state.step = 3
            st.rerun()

if st.session_state.step == 3:
    st.header("Etapa 3: Definição de Regras de Matching e Chaves")
    st.session_state.chave_origem = st.multiselect("Selecione a(s) chave(s) de busca na origem", st.session_state.df_origem.columns)
    st.session_state.chave_destino = st.multiselect("Selecione a(s) chave(s) de busca no destino", st.session_state.df_destino.columns)
    
    mapping_config = {"chave_origem": st.session_state.chave_origem, "chave_destino": st.session_state.chave_destino, "mapeamentos": {}}
    
    for col in st.session_state.df_origem.columns:
        if col not in st.session_state.chave_origem:
            with st.expander(f"Configurar '{col}'"):
                option = st.radio(f"O que deseja fazer com '{col}'?", ["Ignorar", "Mapear"], key=f"option_{col}", index=0)
                if option == "Mapear":
                    mapped_cols = st.multiselect(f"Mapear '{col}' para:", list(st.session_state.df_destino.columns), key=f"map_{col}")
                    function = st.selectbox(f"Tipo de Mapeamento para '{col}'", ["Direct Match", "Aggregation", "Conversion"], key=f"func_{col}")
                    transformation = None
                    if function == "Aggregation":
                        transformation = st.selectbox("Tipo de Agregação", ["Sum", "Mean", "Median", "Max", "Min"], key=f"agg_{col}")
                    elif function == "Conversion":
                        transformation = st.text_area("Definir Dicionário de Conversão (JSON)", "{}", key=f"conv_{col}")
                        try:
                            dict_data = json.loads(transformation)
                            dict_df = pd.DataFrame(list(dict_data.items()), columns=["Origem", "Destino"])
                            st.write("Pré-visualização do Dicionário de Conversão:")
                            st.dataframe(dict_df)
                        except json.JSONDecodeError:
                            st.error("O formato do dicionário de conversão não é válido. Certifique-se de que está em JSON correto.")
                    mapping_config["mapeamentos"][col] = {"destinos": mapped_cols, "funcao": function, "transformacao": transformation}
    
    st.session_state.mapping = mapping_config
    
    if st.checkbox("Mostrar Configuração de Matching (JSON)", value=False, key="show_json"):
        st.json(st.session_state.mapping)
    
    if st.button("Próxima Etapa"):
        st.session_state.step = 4
        st.rerun()

if st.session_state.step == 4:
    st.header("Etapa 4: Execução do Matching")
    if st.session_state.chave_origem and st.session_state.chave_destino:
        st.write("Processo de Matching em execução...")
        executar_matching()
    else:
        st.warning("Defina uma chave de busca válida antes de executar a verificação.")
