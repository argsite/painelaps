
import streamlit as st
import pandas as pd
import plotly.express as px
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(layout="wide", page_title="Dashboard APS - Hipertensão e Diabetes")
st.title("Dashboard APS - Hipertensão e Diabetes")
st.caption("Painel para acompanhamento territorial, busca ativa e monitoramento por equipe e microárea.")


def carregar_planilha(uploaded_file):
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)


def normalizar_colunas(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def faixa_etaria(idade):
    try:
        idade = int(idade)
    except:
        return "Ignorado"
    if idade < 18:
        return "0-17"
    if idade < 40:
        return "18-39"
    if idade < 60:
        return "40-59"
    if idade < 80:
        return "60-79"
    return "80+"


def multiselect_opcoes(label, opcoes):
    opcoes = [o for o in opcoes if pd.notna(o)]
    opcoes = sorted(pd.Series(opcoes).astype(str).unique().tolist())
    return st.sidebar.multiselect(label, opcoes)


def aplicar_filtros_base(df, col_equipe, col_micro, col_prioridade):
    equipes = multiselect_opcoes("Equipe Área", df[col_equipe].dropna().tolist()) if col_equipe in df.columns else []
    micros = multiselect_opcoes("Microárea", df[col_micro].dropna().tolist()) if col_micro in df.columns else []
    faixas = st.sidebar.multiselect("Faixa etária", ["0-17", "18-39", "40-59", "60-79", "80+"])
    prioridades = st.sidebar.multiselect("Prioridade", ["Alta", "Média", "Baixa"])

    filtrado = df.copy()
    if equipes:
        filtrado = filtrado[filtrado[col_equipe].astype(str).isin(equipes)]
    if micros:
        filtrado = filtrado[filtrado[col_micro].astype(str).isin(micros)]
    if faixas and "Faixa Etária" in filtrado.columns:
        filtrado = filtrado[filtrado["Faixa Etária"].isin(faixas)]
    if prioridades and col_prioridade in filtrado.columns:
        filtrado = filtrado[filtrado[col_prioridade].isin(prioridades)]
    return filtrado


def exibir_metricas(cards):
    cols = st.columns(len(cards))
    for col, (titulo, valor) in zip(cols, cards):
        col.metric(titulo, valor)


def grafico_barras(df, x, y, titulo, cor=None):
    if df.empty:
        st.info("Sem dados para exibir neste gráfico.")
        return
    fig = px.bar(df, x=x, y=y, title=titulo, color=cor)
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)


def converter_enderecos(df, col_endereco, col_lat, col_lon):
    geolocator = Nominatim(user_agent="dashboard_aps_porto_feliz")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    df = df.copy()
    enderecos = df[col_endereco].fillna("").astype(str).str.strip()

    if col_lat not in df.columns:
        df[col_lat] = pd.NA
    if col_lon not in df.columns:
        df[col_lon] = pd.NA

    faltantes = df[df[col_lat].isna() | df[col_lon].isna()].index.tolist()
    progresso = st.progress(0)
    status = st.empty()

    total = len(faltantes)
    for i, idx in enumerate(faltantes, start=1):
        endereco = enderecos.loc[idx]
        if not endereco:
            continue
        consulta = geocode(f"{endereco}, Brasil")
        if consulta:
            df.at[idx, col_lat] = consulta.latitude
            df.at[idx, col_lon] = consulta.longitude
        progresso.progress(i / total if total else 1)
        status.write(f"Geocodificando {i} de {total} endereços...")

    status.write("Conversão de endereços concluída.")
    return df


def render_mapa(df, titulo_secao):
    st.subheader(f"Mapa territorial - {titulo_secao}")
    st.markdown("Selecione as colunas para montar o mapa e, se necessário, converta os endereços em latitude/longitude antes da geração.")

    cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    with c1:
        nome_col = st.selectbox("Coluna do nome do paciente", cols, index=cols.index("Nome Completo") if "Nome Completo" in cols else 0, key=f"nome_{titulo_secao}")
    with c2:
        endereco_col = st.selectbox("Coluna do endereço", cols, index=cols.index("Endereço") if "Endereço" in cols else 0, key=f"end_{titulo_secao}")
    with c3:
        area_col = st.selectbox("Coluna da microárea", cols, index=cols.index("Microárea") if "Microárea" in cols else 0, key=f"micro_{titulo_secao}")

    c4, c5 = st.columns(2)
    with c4:
        lat_col = st.selectbox("Coluna de Latitude", [None] + cols, key=f"lat_{titulo_secao}")
    with c5:
        lon_col = st.selectbox("Coluna de Longitude", [None] + cols, key=f"lon_{titulo_secao}")

    sugestao_lat = "Latitude"
    sugestao_lon = "Longitude"

    with st.expander("Converter endereços em latitude e longitude"):
        nova_lat = st.text_input("Nome da coluna de latitude a criar/usar", value=sugestao_lat, key=f"nova_lat_{titulo_secao}")
        nova_lon = st.text_input("Nome da coluna de longitude a criar/usar", value=sugestao_lon, key=f"nova_lon_{titulo_secao}")
        if st.button("Converter endereços", key=f"conv_{titulo_secao}"):
            if not endereco_col:
                st.error("Selecione a coluna de endereço antes de converter.")
            else:
                try:
                    df_convertido = converter_enderecos(df, endereco_col, nova_lat, nova_lon)
                    st.session_state[f"df_mapa_{titulo_secao}"] = df_convertido
                    st.success("Endereços convertidos. Agora selecione as novas colunas de latitude e longitude.")
                    st.dataframe(df_convertido[[c for c in [nome_col, endereco_col, area_col, nova_lat, nova_lon] if c in df_convertido.columns]].head(20), use_container_width=True)
                except Exception as e:
                    st.error(f"Erro na conversão de endereços: {e}")

    df_mapa = st.session_state.get(f"df_mapa_{titulo_secao}", df)

    if st.button("Gerar mapa", key=f"gerar_{titulo_secao}"):
        if not lat_col or not lon_col:
            st.error("Selecione as colunas de latitude e longitude.")
            return
        if lat_col not in df_mapa.columns or lon_col not in df_mapa.columns:
            st.error("As colunas de latitude/longitude selecionadas não existem na tabela atual.")
            return

        dados = df_mapa.copy()
        dados[lat_col] = pd.to_numeric(dados[lat_col], errors="coerce")
        dados[lon_col] = pd.to_numeric(dados[lon_col], errors="coerce")
        dados = dados.dropna(subset=[lat_col, lon_col])

        if dados.empty:
            st.error("Nenhum registro com latitude e longitude válidas foi encontrado.")
            return

        centro = [dados[lat_col].mean(), dados[lon_col].mean()]
        mapa = folium.Map(location=centro, zoom_start=13)
        cores = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "darkgreen"]

        for _, row in dados.iterrows():
            cor = "gray"
            if area_col and area_col in dados.columns:
                cor = cores[hash(str(row.get(area_col, ""))) % len(cores)]
            popup = f"Paciente: {row.get(nome_col, 'N/A')}<br>Endereço: {row.get(endereco_col, 'N/A')}<br>Microárea: {row.get(area_col, 'N/A')}"
            tooltip = f"{row.get(nome_col, 'N/A')} - {row.get(area_col, 'N/A')}"
            folium.Marker(
                location=[row[lat_col], row[lon_col]],
                popup=popup,
                tooltip=tooltip,
                icon=folium.Icon(color=cor),
            ).add_to(mapa)

        st.success("Mapa gerado com sucesso.")
        st_folium(mapa, width=None, height=650)


def preparar_diabetes(df):
    df = normalizar_colunas(df)
    mapa = {
        "nome": "Nome Completo",
        "idade": "Idade",
        "endereco": "Endereço",
        "equipe": "Equipe Área",
        "micro": "Microárea",
        "cadastro": "Cadastro Atualizado",
        "data_cadastro": "Data Atualização Cadastro",
        "consulta": "Consulta Médica/Enfermagem",
        "pa": "Aferição de PA",
        "peso": "Qtd. Registros de peso/altura",
        "visitas": "Qtd. Visitas Domiciliares",
        "hba1c": "Hemoglobina Glicada",
        "pes": "Avaliação dos pés",
        "acomp": "Acompanhado",
    }
    for chave in ["cadastro", "consulta", "pa", "hba1c", "pes", "acomp"]:
        if mapa[chave] in df.columns:
            df[mapa[chave]] = df[mapa[chave]].astype(str).str.upper().str.strip()
    if mapa["idade"] in df.columns:
        df["Faixa Etária"] = df[mapa["idade"]].apply(faixa_etaria)
    df["Sem consulta"] = df[mapa["consulta"]] == "N"
    df["Sem PA"] = df[mapa["pa"]] == "N"
    df["Sem HbA1c"] = df[mapa["hba1c"]] == "N"
    df["Sem avaliação dos pés"] = df[mapa["pes"]] == "N"
    df["Não acompanhado"] = df[mapa["acomp"]] == "N"
    df["Cadastro desatualizado"] = df[mapa["cadastro"]] == "N"
    df["Sem visita"] = pd.to_numeric(df[mapa["visitas"]], errors="coerce").fillna(0) == 0
    df["Pontuação Prioridade"] = (
        df["Sem consulta"].astype(int)
        + df["Sem PA"].astype(int)
        + df["Sem HbA1c"].astype(int)
        + df["Sem avaliação dos pés"].astype(int)
        + df["Não acompanhado"].astype(int)
    )
    df["Prioridade"] = df["Pontuação Prioridade"].map(lambda x: "Alta" if x >= 3 else ("Média" if x == 2 else "Baixa"))
    return df, mapa


def preparar_hipertensao(df):
    df = normalizar_colunas(df)
    mapa = {
        "nome": "Nome Completo",
        "idade": "Idade",
        "endereco": "Endereço",
        "equipe": "Equipe Área",
        "micro": "Microárea",
        "cadastro": "Cadastro Atualizado",
        "data_cadastro": "Data Atualização Cadastro",
        "consulta": "Consulta Médica/Enfermagem",
        "peso": "Qtd. Registros de peso/altura",
        "visitas": "Qtd. Visitas Domiciliares",
        "pa": "Aferição de pressão arterial",
        "acomp": "Acompanhado",
    }
    for chave in ["cadastro", "consulta", "pa", "acomp"]:
        if mapa[chave] in df.columns:
            df[mapa[chave]] = df[mapa[chave]].astype(str).str.upper().str.strip()
    if mapa["idade"] in df.columns:
        df["Faixa Etária"] = df[mapa["idade"]].apply(faixa_etaria)
    df["Sem consulta"] = df[mapa["consulta"]] == "N"
    df["Sem PA"] = df[mapa["pa"]] == "N"
    df["Não acompanhado"] = df[mapa["acomp"]] == "N"
    df["Cadastro desatualizado"] = df[mapa["cadastro"]] == "N"
    df["Sem visita"] = pd.to_numeric(df[mapa["visitas"]], errors="coerce").fillna(0) == 0
    df["Pontuação Prioridade"] = (
        df["Sem consulta"].astype(int)
        + df["Sem PA"].astype(int)
        + df["Não acompanhado"].astype(int)
        + df["Cadastro desatualizado"].astype(int)
        + df["Sem visita"].astype(int)
    )
    df["Prioridade"] = df["Pontuação Prioridade"].map(lambda x: "Alta" if x >= 3 else ("Média" if x == 2 else "Baixa"))
    return df, mapa


def render_diabetes(df):
    df, m = preparar_diabetes(df)
    st.sidebar.header("Filtros")
    filtrado = aplicar_filtros_base(df, m["equipe"], m["micro"], "Prioridade")
    total = len(filtrado)
    pct = lambda col: f"{((filtrado[col] == 'S').mean() * 100 if total else 0):.1f}%"
    pct_visita = f"{(((pd.to_numeric(filtrado[m['visitas']], errors='coerce').fillna(0) > 0).mean() * 100) if total else 0):.1f}%"
    exibir_metricas([
        ("Total", total),
        ("Consulta", pct(m["consulta"])),
        ("PA", pct(m["pa"])),
        ("HbA1c", pct(m["hba1c"])),
        ("Pés", pct(m["pes"])),
        ("Acompanhados", pct(m["acomp"])),
        ("Com visita", pct_visita),
    ])
    c1, c2 = st.columns(2)
    with c1:
        g1 = filtrado.groupby(m["micro"], dropna=False)["Não acompanhado"].sum().reset_index().sort_values("Não acompanhado", ascending=False)
        grafico_barras(g1, m["micro"], "Não acompanhado", "Não acompanhados por microárea")
    with c2:
        pend = pd.DataFrame({
            "Indicador": ["Sem consulta", "Sem PA", "Sem HbA1c", "Sem avaliação dos pés"],
            "Quantidade": [
                int(filtrado["Sem consulta"].sum()),
                int(filtrado["Sem PA"].sum()),
                int(filtrado["Sem HbA1c"].sum()),
                int(filtrado["Sem avaliação dos pés"].sum()),
            ],
        })
        grafico_barras(pend, "Indicador", "Quantidade", "Pendências do cuidado - Diabetes")
    st.subheader("Lista nominal para busca ativa")
    cols = [m["nome"], m["idade"], m["endereco"], m["equipe"], m["micro"], m["consulta"], m["pa"], m["hba1c"], m["pes"], m["visitas"], m["acomp"], "Prioridade"]
    st.dataframe(filtrado[[c for c in cols if c in filtrado.columns]], use_container_width=True)
    render_mapa(filtrado, "Diabetes")


def render_hipertensao(df):
    df, m = preparar_hipertensao(df)
    st.sidebar.header("Filtros")
    filtrado = aplicar_filtros_base(df, m["equipe"], m["micro"], "Prioridade")
    total = len(filtrado)
    pct = lambda col: f"{((filtrado[col] == 'S').mean() * 100 if total else 0):.1f}%"
    pct_visita = f"{(((pd.to_numeric(filtrado[m['visitas']], errors='coerce').fillna(0) > 0).mean() * 100) if total else 0):.1f}%"
    exibir_metricas([
        ("Total", total),
        ("Consulta", pct(m["consulta"])),
        ("PA aferida", pct(m["pa"])),
        ("Com visita", pct_visita),
        ("Cadastro atualizado", pct(m["cadastro"])),
        ("Acompanhados", pct(m["acomp"])),
    ])
    c1, c2 = st.columns(2)
    with c1:
        g1 = filtrado.groupby(m["micro"], dropna=False)["Sem PA"].sum().reset_index().sort_values("Sem PA", ascending=False)
        grafico_barras(g1, m["micro"], "Sem PA", "Sem aferição de PA por microárea")
    with c2:
        pend = pd.DataFrame({
            "Indicador": ["Sem consulta", "Sem PA", "Sem visita", "Não acompanhado", "Cadastro desatualizado"],
            "Quantidade": [
                int(filtrado["Sem consulta"].sum()),
                int(filtrado["Sem PA"].sum()),
                int(filtrado["Sem visita"].sum()),
                int(filtrado["Não acompanhado"].sum()),
                int(filtrado["Cadastro desatualizado"].sum()),
            ],
        })
        grafico_barras(pend, "Indicador", "Quantidade", "Pendências do cuidado - Hipertensão")
    st.subheader("Lista nominal para busca ativa")
    cols = [m["nome"], m["idade"], m["endereco"], m["equipe"], m["micro"], m["consulta"], m["pa"], m["visitas"], m["acomp"], "Prioridade"]
    st.dataframe(filtrado[[c for c in cols if c in filtrado.columns]], use_container_width=True)
    render_mapa(filtrado, "Hipertensão")


secao = st.sidebar.radio("Linha de cuidado", ["Hipertensão", "Diabetes"])
arquivo = st.file_uploader("Envie a planilha correspondente", type=["xlsx", "xls", "csv"])

with st.expander("Como usar"):
    st.markdown("""
    - Escolha a linha de cuidado na barra lateral.
    - Envie a planilha correspondente daquela seção.
    - Use os filtros por equipe, microárea, faixa etária e prioridade.
    - A tabela nominal ajuda na organização da busca ativa.
    - Na parte do mapa, você pode selecionar nome, endereço, microárea, latitude e longitude.
    - Se a planilha não tiver coordenadas, use a opção de converter os endereços antes de gerar o mapa.
    """)

if arquivo is None:
    st.info("Aguardando upload da planilha de hipertensão ou diabetes.")
else:
    try:
        df = carregar_planilha(arquivo)
        if secao == "Hipertensão":
            render_hipertensao(df)
        else:
            render_diabetes(df)
    except Exception as e:
        st.error(f"Não foi possível processar a planilha: {e}")
