
import streamlit as st
import pandas as pd
from io import BytesIO
import plotly.express as px
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(layout="wide", page_title="Dashboard APS - Hipertensão e Diabetes")
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    padding: 14px 16px;
    border-radius: 14px;
    min-height: 112px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
div[data-testid="metric-container"] label {
    white-space: normal !important;
    overflow: visible !important;
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    white-space: normal !important;
    overflow: visible !important;
}
div[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
    font-size: 0.95rem !important;
    line-height: 1.2 !important;
    min-height: 2.4em;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 2rem !important;
    line-height: 1.1 !important;
}
</style>
""", unsafe_allow_html=True)
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
    n = len(cards)
    if n <= 4:
        cols = st.columns(n)
    else:
        cols = st.columns(3)
    for i, (titulo, valor) in enumerate(cards):
        cols[i % len(cols)].metric(titulo, valor)


def grafico_barras(df, x, y, titulo, cor=None):
    if df.empty:
        st.info("Sem dados para exibir neste gráfico.")
        return
    fig = px.bar(df, x=x, y=y, title=titulo, color=cor)
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)


def normalizar_visitas(serie):
    s = serie.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "N/A": pd.NA, "-": pd.NA})
    s = s.str.replace(",", ".", regex=False)
    s = s.replace({"2+": 2.1})
    return pd.to_numeric(s, errors="coerce").fillna(0)


def montar_motivo_busca_ativa(row, linha_cuidado):
    motivos = []
    if bool(row.get("Sem consulta", False)):
        motivos.append("Sem consulta")
    if bool(row.get("Sem PA", False)):
        motivos.append("Sem PA")
    if linha_cuidado == "Diabetes" and bool(row.get("Sem HbA1c", False)):
        motivos.append("Sem HbA1c")
    if linha_cuidado == "Diabetes" and bool(row.get("Sem avaliação dos pés", False)):
        motivos.append("Sem avaliação dos pés")
    if bool(row.get("Sem visita", False)):
        motivos.append("Sem visita")
    if bool(row.get("Não acompanhado", False)):
        motivos.append("Não acompanhado")
    if bool(row.get("Cadastro desatualizado", False)):
        motivos.append("Cadastro desatualizado")
    return " + ".join(motivos) if motivos else "Sem pendências críticas"


def sugerir_acao(row, linha_cuidado):
    if bool(row.get("Sem visita", False)) and bool(row.get("Não acompanhado", False)):
        return "Realizar visita domiciliar e articular acompanhamento da equipe"
    if bool(row.get("Sem consulta", False)) and bool(row.get("Sem PA", False)):
        return "Priorizar avaliação clínica e verificação de PA"
    if linha_cuidado == "Diabetes" and (bool(row.get("Sem HbA1c", False)) or bool(row.get("Sem avaliação dos pés", False))):
        return "Programar cuidado do diabetes e atualizar exames/avaliações"
    if bool(row.get("Cadastro desatualizado", False)):
        return "Atualizar cadastro no próximo contato"
    if bool(row.get("Não acompanhado", False)):
        return "Reinserir no acompanhamento da equipe"
    return "Manter monitoramento de rotina"


def preparar_lista_nominal_inteligente(df, linha_cuidado, m):
    lista = df.copy()
    lista["Motivo da busca ativa"] = lista.apply(lambda row: montar_motivo_busca_ativa(row, linha_cuidado), axis=1)
    lista["Ação sugerida"] = lista.apply(lambda row: sugerir_acao(row, linha_cuidado), axis=1)
    ordenar = [c for c in ["Pontuação Prioridade", m.get("micro"), m.get("nome")] if c in lista.columns]
    asc = [False, True, True][:len(ordenar)]
    if ordenar:
        lista = lista.sort_values(by=ordenar, ascending=asc)
    return lista


def construir_mapa(dados, lat_col, lon_col, nome_col, endereco_col, area_col, modo_mapa="Agrupado"):
    centro = [dados[lat_col].mean(), dados[lon_col].mean()]
    mapa = folium.Map(location=centro, zoom_start=13)
    cores = ["red", "blue", "green", "purple", "orange", "darkred", "cadetblue", "darkgreen"]

    if modo_mapa == "Calor":
        pontos = dados[[lat_col, lon_col]].dropna().values.tolist()
        if pontos:
            HeatMap(pontos, radius=18, blur=14, min_opacity=0.35).add_to(mapa)
        return mapa

    camada = MarkerCluster().add_to(mapa) if modo_mapa == "Agrupado" else mapa

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
        ).add_to(camada)
    return mapa


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




def identificar_linha_cuidado(df):
    cols = [str(c).strip() for c in df.columns]
    if "Hemoglobina Glicada" in cols or "Avaliação dos pés" in cols:
        return "Diabetes"
    if "Aferição de pressão arterial" in cols:
        return "Hipertensão"
    return None

def dataframe_para_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Dados")
    output.seek(0)
    return output.getvalue()


def render_mapa(df, titulo_secao):
    st.subheader(f"Mapa territorial - {titulo_secao}")
    cols = df.columns.tolist()
    df_mapa = st.session_state.get(f"df_mapa_{titulo_secao}", df)

    nome_default = cols.index("Nome Completo") if "Nome Completo" in cols else 0
    endereco_default = cols.index("Endereço") if "Endereço" in cols else 0
    area_default = cols.index("Microárea") if "Microárea" in cols else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        nome_col = st.selectbox("Nome do paciente", cols, index=nome_default, key=f"nome_{titulo_secao}")
    with c2:
        endereco_col = st.selectbox("Endereço", cols, index=endereco_default, key=f"end_{titulo_secao}")
    with c3:
        area_col = st.selectbox("Microárea", cols, index=area_default, key=f"micro_{titulo_secao}")

    lat_options = ["Latitude"] + [c for c in cols if c != "Latitude"]
    lon_options = ["Longitude"] + [c for c in cols if c != "Longitude"]
    lat_col = st.selectbox("Latitude", lat_options, index=0 if "Latitude" in lat_options else None, key=f"lat_{titulo_secao}")
    lon_col = st.selectbox("Longitude", lon_options, index=0 if "Longitude" in lon_options else None, key=f"lon_{titulo_secao}")

    c4, c5, c6 = st.columns([1, 2, 1.3])
    with c4:
        converter = st.checkbox("Converter endereços automaticamente", key=f"check_conv_{titulo_secao}")
    with c5:
        st.caption("Use essa opção só se a planilha ainda não tiver latitude e longitude.")
    with c6:
        modo_mapa = st.selectbox("Tipo de mapa", ["Agrupado", "Pontos", "Calor"], key=f"modo_{titulo_secao}")

    if converter:
        if st.button("Converter e preparar mapa", key=f"conv_{titulo_secao}"):
            try:
                df_convertido = converter_enderecos(df_mapa, endereco_col, "Latitude", "Longitude")
                st.session_state[f"df_mapa_{titulo_secao}"] = df_convertido
                st.success("Conversão concluída. As colunas Latitude e Longitude foram preparadas.")
                df_mapa = df_convertido
            except Exception as e:
                st.error(f"Erro na conversão de endereços: {e}")

    df_mapa = st.session_state.get(f"df_mapa_{titulo_secao}", df_mapa)

    if "Latitude" in df_mapa.columns and "Longitude" in df_mapa.columns:
        arquivo_excel = dataframe_para_excel_bytes(df_mapa)
        st.download_button(
            label="Baixar planilha geocodificada",
            data=arquivo_excel,
            file_name=f"{titulo_secao.lower()}_geocodificada.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_geo_{titulo_secao}",
        )

    if st.button("Gerar mapa", key=f"gerar_{titulo_secao}"):
        if lat_col not in df_mapa.columns or lon_col not in df_mapa.columns:
            st.error("Não encontrei as colunas de latitude e longitude. Marque a opção de conversão ou selecione colunas válidas.")
        else:
            dados = df_mapa.copy()
            dados[lat_col] = pd.to_numeric(dados[lat_col], errors="coerce")
            dados[lon_col] = pd.to_numeric(dados[lon_col], errors="coerce")
            dados = dados.dropna(subset=[lat_col, lon_col])

            if dados.empty:
                st.error("Nenhum registro com latitude e longitude válidas foi encontrado.")
            else:
                st.session_state[f"dados_mapa_{titulo_secao}"] = dados
                st.session_state[f"config_mapa_{titulo_secao}"] = {
                    "lat_col": lat_col,
                    "lon_col": lon_col,
                    "nome_col": nome_col,
                    "endereco_col": endereco_col,
                    "area_col": area_col,
                    "modo_mapa": modo_mapa,
                }
                st.success("Mapa gerado com sucesso.")

    dados_salvos = st.session_state.get(f"dados_mapa_{titulo_secao}")
    cfg = st.session_state.get(f"config_mapa_{titulo_secao}")
    if dados_salvos is not None and cfg is not None and not dados_salvos.empty:
        mapa = construir_mapa(
            dados_salvos,
            cfg["lat_col"],
            cfg["lon_col"],
            cfg["nome_col"],
            cfg["endereco_col"],
            cfg["area_col"],
            cfg.get("modo_mapa", "Agrupado"),
        )
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
    df["Visitas Normalizadas"] = normalizar_visitas(df[mapa["visitas"]])
    df["Sem visita"] = df["Visitas Normalizadas"] <= 0
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
    df["Visitas Normalizadas"] = normalizar_visitas(df[mapa["visitas"]])
    df["Sem visita"] = df["Visitas Normalizadas"] <= 0
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
    pct_visita = f"{(((filtrado['Visitas Normalizadas'] > 0).mean() * 100) if total else 0):.1f}%"
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
    lista_inteligente = preparar_lista_nominal_inteligente(filtrado, "Diabetes", m)
    somente_criticos = st.checkbox("Mostrar apenas casos com pendências", value=False, key="criticos_diabetes")
    if somente_criticos:
        lista_inteligente = lista_inteligente[lista_inteligente["Pontuação Prioridade"] > 0]
    cols = [m["nome"], m["idade"], m["endereco"], m["equipe"], m["micro"], "Prioridade", "Pontuação Prioridade", "Motivo da busca ativa", "Ação sugerida", m["consulta"], m["pa"], m["hba1c"], m["pes"], m["visitas"], m["acomp"]]
    st.dataframe(lista_inteligente[[c for c in cols if c in lista_inteligente.columns]], use_container_width=True)
    render_mapa(filtrado, "Diabetes")


def render_hipertensao(df):
    df, m = preparar_hipertensao(df)
    st.sidebar.header("Filtros")
    filtrado = aplicar_filtros_base(df, m["equipe"], m["micro"], "Prioridade")
    total = len(filtrado)
    pct = lambda col: f"{((filtrado[col] == 'S').mean() * 100 if total else 0):.1f}%"
    pct_visita = f"{(((filtrado['Visitas Normalizadas'] > 0).mean() * 100) if total else 0):.1f}%"
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
    lista_inteligente = preparar_lista_nominal_inteligente(filtrado, "Hipertensão", m)
    somente_criticos = st.checkbox("Mostrar apenas casos com pendências", value=False, key="criticos_hipertensao")
    if somente_criticos:
        lista_inteligente = lista_inteligente[lista_inteligente["Pontuação Prioridade"] > 0]
    cols = [m["nome"], m["idade"], m["endereco"], m["equipe"], m["micro"], "Prioridade", "Pontuação Prioridade", "Motivo da busca ativa", "Ação sugerida", m["consulta"], m["pa"], m["visitas"], m["acomp"]]
    st.dataframe(lista_inteligente[[c for c in cols if c in lista_inteligente.columns]], use_container_width=True)
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
