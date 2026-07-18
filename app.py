import streamlit as st
import pandas as pd
import plotly.express as px

# 1. CONFIGURAÇÃO CENTRALIZADA (Adicione ou ajuste conforme necessário)
# O padrão interno do dashboard será: 'nome', 'cns', 'equipe', 'status'
CONFIG = {
    "Diabetes": {"colunas": ["HbA1c", "Pés"]},
    "Gestação": {"colunas": ["Pré-natal", "dTpa"]},
    "Infantil": {"colunas": ["Consulta 1º mês", "Pentavalente"]},
    "Hipertensão": {"colunas": ["PA"]},
    "Idoso": {"colunas": ["Influenza"]},
    "Câncer": {"colunas": ["Rast. Colo", "Rast. Mama"]}
}

# 2. FUNÇÃO DE LEITURA RESILIENTE (Resolve o erro de codificação)
def carregar_dados(uploaded_file):
    """Tenta ler o arquivo com diferentes codificações."""
    for encoding in ['latin-1', 'utf-8', 'cp1252']:
        try:
            # Tenta CSV primeiro
            return pd.read_csv(uploaded_file, encoding=encoding, sep=',')
        except:
            continue
    
    # Se falhar, tenta Excel (arquivos .xls renomeados como .csv)
    try:
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Não foi possível ler o arquivo: {e}")
        return None

# 3. FUNÇÃO DE NORMALIZAÇÃO
def normalizar_df(df):
    """Padroniza nomes de colunas cruciais."""
    mapeamento = {
        'Nome Completo': 'nome',
        'CNS': 'cns',
        'Equipe Área': 'equipe',
        'Acompanhado': 'status'
    }
    return df.rename(columns=mapeamento)

# 4. INTERFACE PRINCIPAL
st.set_page_config(layout="wide", page_title="Dashboard Saúde 360")
st.title("Dashboard APS - Saúde 360")

tipo_selecionado = st.sidebar.selectbox("Selecione o Indicador", list(CONFIG.keys()))
uploaded_file = st.sidebar.file_uploader(f"Upload - {tipo_selecionado}", type=["csv", "xls", "xlsx"])

if uploaded_file:
    df = carregar_dados(uploaded_file)
    if df is not None:
        df_processado = normalizar_df(df)
        
        st.success(f"Dados de {tipo_selecionado} carregados com sucesso!")
        
        # Exemplo de visualização modular
        c1, c2 = st.columns(2)
        c1.metric("Total de Pacientes", len(df_processado))
        
        # Filtro de status se existir
        if 'status' in df_processado.columns:
            acompanhados = df_processado['status'].value_counts().get('S', 0)
            c2.metric("Acompanhados", acompanhados)
        
        st.dataframe(df_processado.head())
        
        # Gráfico dinâmico por equipe
        if 'equipe' in df_processado.columns:
            fig = px.bar(df_processado['equipe'].value_counts(), title="Atendimento por Equipe")
            st.plotly_chart(fig, use_container_width=True)

[Image of a data ingestion pipeline handling different file encoding formats]
