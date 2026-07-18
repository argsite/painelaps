import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. CONFIGURAÇÃO (Adicione todas as colunas que deseja monitorar)
CONFIG = {
    "Diabetes": {
        "pendencias": ["Sem HbA1c", "Sem avaliação dos pés", "Sem Consulta", "Sem Visitas"]
    },
    "Gestação": {"pendencias": ["Sem pré-natal", "Sem dTpa"]},
    "Infantil": {"pendencias": ["Sem consulta", "Sem Pentavalente"]},
    "Hipertensão": {"pendencias": ["Sem PA"]},
    "Idoso": {"pendencias": ["Sem Vacina Influenza"]},
    "Câncer": {"pendencias": ["Sem Rast. Colo", "Sem Rast. Mama"]}
}

def carregar_dados(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    try:
        uploaded_file.seek(0)
        if ext in ['.xls', '.xlsx']: return pd.read_excel(uploaded_file)
        for enc in ['latin-1', 'utf-8', 'cp1252']:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding=enc, sep=None, engine='python')
            except: continue
    except: return None

def processar_df(df, tipo):
    # Mapeamento padrão
    mapeamento = {'Nome Completo': 'nome', 'CNS': 'cns', 'Equipe Área': 'equipe', 
                  'Microárea': 'micro', 'Idade': 'idade', 'Acompanhado': 'status'}
    df = df.rename(columns=mapeamento)
    
    # Lógica para colunas de Consulta/Visitas (Numéricas)
    # Se o nome da coluna no seu Excel contiver "Consulta" ou "Visitas", tratamos como número
    for col in df.columns:
        if any(x in col for x in ["Consulta", "Visitas"]):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('+', ''), errors='coerce').fillna(0)
            # Cria a coluna "Sem..." se ela não existir
            col_pend = f"Sem {col.replace('Sem ', '')}"
            df[col_pend] = df[col].apply(lambda x: 'N' if x == 0 else 'S')
            
    # Garantir colunas padrão
    for col in ['micro', 'idade', 'status', 'equipe']:
        if col not in df.columns: df[col] = 'N/A'
    
    # Calcular total de pendências
    cols_pend = CONFIG[tipo]["pendencias"]
    cols_existentes = [c for c in cols_pend if c in df.columns]
    df['Total Pendências'] = df[cols_existentes].eq('N').sum(axis=1) if cols_existentes else 0
    return df

# 4. INTERFACE
st.set_page_config(layout="wide", page_title="Dashboard Saúde 360")
st.title("Dashboard APS - Saúde 360")

tipo_sel = st.sidebar.selectbox("Indicador", list(CONFIG.keys()))
uploaded_file = st.sidebar.file_uploader("Upload", type=["csv", "xls", "xlsx"])

if uploaded_file:
    df_base = processar_df(carregar_dados(uploaded_file), tipo_sel)
    
    # --- FILTROS ---
    st.sidebar.markdown("### 🔍 Filtros de Busca Ativa")
    df_filtrado = df_base.copy()
    
    # Filtros de Categorias
    for col in ['equipe', 'micro', 'idade']:
        if col in df_filtrado.columns:
            sel = st.sidebar.multiselect(col.capitalize(), sorted(df_base[col].astype(str).unique()))
            if sel: df_filtrado = df_filtrado[df_filtrado[col].isin(sel)]
            
    # Filtro de Pendências (Dinâmico)
    pend_existentes = [p for p in CONFIG[tipo_sel]["pendencias"] if p in df_base.columns]
    sel_pend = st.sidebar.multiselect("Pendências", pend_existentes)
    if sel_pend:
        df_filtrado = df_filtrado[df_filtrado[sel_pend].eq('N').any(axis=1)]
        
    # --- PAINEL ---
    st.subheader("📊 Visão Geral (Base Completa)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total de Pacientes", len(df_base))
    c2.metric("Acompanhados", df_base[df_base['status'] == 'S'].shape[0])
    c3.metric("Média de Pendências", f"{df_base['Total Pendências'].mean():.1f}")
    
    st.divider()
    st.subheader(f"📋 Busca Ativa ({len(df_filtrado)} pacientes)")
    st.dataframe(df_filtrado.sort_values(by='Total Pendências', ascending=False), use_container_width=True)
    
    csv = df_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button("Baixar Lista Filtrada", csv, "busca_ativa.csv", "text/csv")
