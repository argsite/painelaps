import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. CONFIGURAÇÃO CENTRALIZADA
CONFIG = {
    "Diabetes": {"pendencias": ["Sem HbA1c", "Sem avaliação dos pés"]},
    "Gestação": {"pendencias": ["Sem pré-natal", "Sem dTpa"]},
    "Infantil": {"pendencias": ["Sem consulta", "Sem Pentavalente"]},
    "Hipertensão": {"pendencias": ["Sem PA"]},
    "Idoso": {"pendencias": ["Sem Vacina Influenza"]},
    "Câncer": {"pendencias": ["Sem Rast. Colo", "Sem Rast. Mama"]}
}

# 2. FUNÇÃO DE LEITURA RESILIENTE
def carregar_dados(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        uploaded_file.seek(0)
        if ext in ['.xls', '.xlsx']:
            return pd.read_excel(uploaded_file)
        else:
            for enc in ['latin-1', 'utf-8', 'cp1252', 'iso-8859-1']:
                try:
                    uploaded_file.seek(0)
                    return pd.read_csv(uploaded_file, encoding=enc, sep=None, engine='python')
                except: continue
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        return None

# 3. NORMALIZAÇÃO E PRIORIZAÇÃO
def processar_df(df, tipo):
    mapeamento = {
        'Nome Completo': 'nome', 'CNS': 'cns', 
        'Equipe Área': 'equipe', 'Acompanhado': 'status'
    }
    df = df.rename(columns=mapeamento)
    
    # Cria coluna de "Prioridade" (contagem de 'N' nas colunas configuradas)
    cols_pend = CONFIG[tipo]["pendencias"]
    cols_existentes = [c for c in cols_pend if c in df.columns]
    
    if cols_existentes:
        df['Total Pendências'] = df[cols_existentes].eq('N').sum(axis=1)
    else:
        df['Total Pendências'] = 0
    return df

# 4. INTERFACE
st.set_page_config(layout="wide", page_title="Dashboard Saúde 360")
st.title("Dashboard APS - Saúde 360")

tipo_selecionado = st.sidebar.selectbox("Selecione o Indicador", list(CONFIG.keys()))
uploaded_file = st.sidebar.file_uploader("Upload de Relatório", type=["csv", "xls", "xlsx"])

if uploaded_file:
    df_raw = carregar_dados(uploaded_file)
    if df_raw is not None:
        df = processar_df(df_raw, tipo_selecionado)
        
        # --- FILTROS SIDEBAR COM SEGURANÇA ---
        st.sidebar.markdown("### 🔍 Filtros de Busca Ativa")
        
        if 'equipe' in df.columns:
            equipes = sorted(df['equipe'].astype(str).unique())
            equipes_sel = st.sidebar.multiselect("Filtrar por Equipe", equipes)
            if equipes_sel: df = df[df['equipe'].isin(equipes_sel)]
            
        pendencias_config = CONFIG[tipo_selecionado]["pendencias"]
        pendencias_existentes = [p for p in pendencias_config if p in df.columns]
        
        if pendencias_existentes:
            pendencias_sel = st.sidebar.multiselect("Filtrar Pendências", pendencias_existentes)
            if pendencias_sel:
                df = df[df[pendencias_sel].eq('N').any(axis=1)]
        
        # --- DASHBOARD REATIVO ---
        st.info(f"Visualizando {len(df)} pacientes após os filtros.")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total de Pacientes", len(df))
        c2.metric("Acompanhados", df[df['status'] == 'S'].shape[0] if 'status' in df.columns else 0)
        c3.metric("Média de Pendências", f"{df['Total Pendências'].mean():.1f}")
        
        tab1, tab2 = st.tabs(["📊 Painel de Desempenho", "📋 Lista Nominal"])
        
        with tab1:
            if 'equipe' in df.columns:
                fig = px.bar(df['equipe'].value_counts().reset_index(), x='index', y='equipe', title="Distribuição por Equipe")
                st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            df_lista = df.sort_values(by='Total Pendências', ascending=False)
            st.dataframe(df_lista, use_container_width=True)
            
            csv = df_lista.to_csv(index=False).encode('utf-8')
            st.download_button("Baixar Lista de Busca Ativa", csv, "lista_busca_ativa.csv", "text/csv")
else:
    st.info("Aguardando o upload do arquivo.")
