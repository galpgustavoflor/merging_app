import streamlit as st
import pandas as pd
import plotly.express as px
import json
import concurrent.futures
import dask.dataframe as dd

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

if 'validation' not in st.session_state:
    st.session_state.business_rules = {}

if 'validation_rules' not in st.session_state:
    st.session_state.validation_rules = {}

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

def executar_matching_dask():
    """
    Executa o processo de matching usando Dask para manipular grandes volumes de dados
    diretamente a partir de um DataFrame Pandas já carregado.
    """
    st.write("Executando matching ...")

    df_origem = st.session_state.df_origem  # Já carregado como Pandas
    df_destino = st.session_state.df_destino  # Já carregado como Pandas

    chave_origem = st.session_state.chave_origem
    chave_destino = st.session_state.chave_destino

    # Converter df_origem para um DataFrame Dask para processar em paralelo
    ddf_origem = dd.from_pandas(df_origem, npartitions=10)  # Divide em 10 partições
    ddf_destino = dd.from_pandas(df_destino, npartitions=10)

    # Realizar o merge usando Dask
    ddf_final = ddf_origem.merge(
        ddf_destino, 
        left_on=chave_origem, 
        right_on=chave_destino, 
        how="outer", 
        indicator=True
    )

    # Exibir resumo do Matching
    total_match = ddf_final[ddf_final['_merge'] == 'both'].shape[0].compute()
    faltantes_origem = ddf_final[ddf_final['_merge'] == 'right_only'].shape[0].compute()
    faltantes_destino = ddf_final[ddf_final['_merge'] == 'left_only'].shape[0].compute()

    st.write("### Resumo do Matching")
    st.write(f"Total de registros correspondentes: {total_match}")
    st.write(f"Total de registros faltantes na origem: {faltantes_origem}")
    st.write(f"Total de registros faltantes no destino: {faltantes_destino}")

    # Converter de volta para Pandas para exibir no Streamlit (somente amostra para evitar problemas de memória)
    df_final_sample = ddf_final.compute().sample(n=min(500, len(ddf_final)), random_state=42)  # Exibir 500 amostras

    st.write("### Registros Correspondentes")
    st.dataframe(df_final_sample[df_final_sample['_merge'] == 'both'])

    st.write("### Registros Não Correspondentes")
    st.dataframe(df_final_sample[df_final_sample['_merge'] != 'both'])

    return ddf_final  # Mantém como Dask para manipulação eficiente

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
        matching_results = executar_matching_dask()
        st.session_state.matching_results = matching_results[matching_results['_merge'] == 'both']
    else:
        st.warning("Defina uma chave de busca válida antes de executar a verificação.")
    
    if st.button("Configurar validações de dados"):
        st.session_state.step = 5
        st.rerun()

if st.session_state.step == 5:
    st.header("Etapa 5: Mapeamento de Regras de Negócio")

    business_rules_config = {}

    st.subheader("Validações Diretas")
    validation_rules = {}
    for col in st.session_state.df_destino.columns:
        with st.popover(f"Configurar validação para '{col}'"):
            validate_nulls = st.checkbox("Verificar Nulos", key=f"nulls_{col}")
            validate_unique = st.checkbox("Verificar Unicidade", key=f"unique_{col}")
            validate_domain = st.checkbox("Verificar Lista de Valores (separados por virgula)", key=f"domain_{col}")
            domain_values = []
            if validate_domain:
                domain_values = st.text_input("Valores Permitidos", key=f"domain_values_{col}")
            validate_regex = st.checkbox("Verificar Formato (Regex)", key=f"regex_{col}")
            regex_pattern = ""
            if validate_regex:
                regex_pattern = st.text_input("Expressão Regular", key=f"regex_pattern_{col}")
            if validate_nulls or validate_unique or validate_domain or validate_regex:
                validation_rules[col] = {}
            if validate_nulls:
                validation_rules[col]["validar_nulos"] = True
            if validate_unique:
                validation_rules[col]["validar_unicidade"] = True
            if validate_domain:
                validation_rules[col]["validar_lista_de_valores"] = domain_values.split(',')
            if validate_regex:
                validation_rules[col]["validar_regex"] = regex_pattern
    
    st.session_state.validation_rules = validation_rules


    if st.checkbox("Mostrar Configuração de Validação (JSON)", value=False, key="show_json"):
        st.json(st.session_state.validation_rules)





    if st.button("Próxima Etapa"):
        st.session_state.step = 6
        st.rerun()

if st.session_state.step == 6:
    st.header("Etapa 6: Validação de Dados")
