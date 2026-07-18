import streamlit as st
import pandas as pd
import plotly.express as px

# 1. CONFIGURAÇÃO CENTRALIZADA
# Adicione ou ajuste as chaves conforme o nome exato das colunas nos seus CSVs
CONFIG = {
    "Diabetes": {
        "colunas_pendencia": ["Sem HbA1c", "Sem avaliação dos pés"],
        "metricas": ["Consulta", "PA", "HbA1c", "Pés"]
    },
    "Gestação": {
        "colunas_pendencia": ["Sem pré-natal", "Sem dTpa"],
        "metricas": ["Consulta Pré-natal", "Testes Rápidos", "dTpa"]
    },
    "Infantil": {
        "colunas_pendencia": ["Sem Pentavalente", "Sem Pólio"],
        "metricas": ["Consulta 1º mês", "Pentavalente", "Pneumocócica"]
    },
    "Hipertensão": {
        "colunas_pendencia": ["Sem PA"],
        "metricas": ["Consulta", "PA", "Visitas"]
    },
    "Idoso": {
        "colunas_pendencia": ["Sem Vacina Influenza"],
        "metricas": ["Consulta Médica", "Visitas", "Vacina Influenza"]
    },
    "Câncer": {
        "colunas_pendencia": ["Sem Rast. Colo", "Sem Rast. Mama"],
        "metricas": ["Rast. Colo", "Rast. Mama", "Vacina HPV"]
    }
}

# 2. FUNÇÃO DE NORMALIZAÇÃO
def processar_dados(df, tipo):
    """Padroniza colunas para um formato genérico para o Dashboard."""
    df = df.copy()
    # Mapeamento genérico para identificação
    mapa_fixo = {
        'Nome Completo': 'nome',
        'CNS': 'cns',
        'Equipe Área': 'equipe',
        'Acompanhado': 'status'
    }
    df = df.rename(columns=mapa_fixo)
    
    # Garantir que colunas de status existam para filtros
    if 'status' not in df.columns:
        df['status'] = 'S' # Default se não encontrado
    return df

# 3. INTERFACE STREAMLIT
st.set_page_config(layout="wide", page_title="Dashboard Saúde 360")
st.title("Dashboard APS - Saúde 360")

# Menu de seleção
tipo_selecionado = st.sidebar.selectbox("Selecione o Indicador", list(CONFIG.keys()))
uploaded_file = st.sidebar.file_uploader(f"Upload de Relatório - {tipo_selecionado}", type=["csv", "xls", "xlsx"])

if uploaded_file:
    # Leitura dinâmica
    try:
        df = pd.read_csv(uploaded_file)
        df_processado = processar_dados(df, tipo_selecionado)
        
        st.subheader(f"Análise de {tipo_selecionado}")
        
        # Métricas Globais
        c1, c2, c3 = st.columns(3)
        c1.metric("Total de Pacientes", len(df_processado))
        c2.metric("Acompanhados", df_processado['status'].value_counts().get('S', 0))
        
        # Filtros e Visualização
        st.write("### Visão Geral por Equipe")
        if 'equipe' in df_processado.columns:
            fig = px.bar(df_processado['equipe'].value_counts(), title="Pacientes por Equipe")
            st.plotly_chart(fig, use_container_width=True)
            
        st.write("### Lista Nominal")
        st.dataframe(df_processado)
        
    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")
else:
    st.info("Por favor, faça o upload do arquivo CSV correspondente ao indicador selecionado.")
