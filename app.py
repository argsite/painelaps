import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(layout="wide", page_title="Mapa de Saúde Territorial")

if "df_geo" not in st.session_state:
    st.session_state.df_geo = None
if "mostrar_mapa" not in st.session_state:
    st.session_state.mostrar_mapa = False
if "area_col_atual" not in st.session_state:
    st.session_state.area_col_atual = None
if "modo_atual" not in st.session_state:
    st.session_state.modo_atual = None

st.title("📍 Dashboard de Saúde - Rastreamento Territorial")
st.markdown("Envie uma planilha com endereços ou coordenadas para visualizar os pacientes no mapa.")

uploaded_file = st.file_uploader("Escolha sua planilha (Excel ou CSV)", type=["xlsx", "csv"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    st.write("### Pré-visualização dos dados")
    st.dataframe(df.head())

    cols = df.columns.tolist()
    modo = st.radio(
        "Como deseja gerar o mapa?",
        ["Usar latitude/longitude já existentes", "Converter endereço em latitude/longitude"]
    )
    st.session_state.modo_atual = modo

    if modo == "Usar latitude/longitude já existentes":
        lat_col = st.selectbox("Selecione a coluna de Latitude", cols, index=None)
        lon_col = st.selectbox("Selecione a coluna de Longitude", cols, index=None)
        area_col = st.selectbox("Selecione a coluna de Microárea (opcional)", [None] + cols)
        st.session_state.area_col_atual = area_col

        if st.button("Gerar mapa com coordenadas existentes"):
            if lat_col and lon_col:
                df_mapa = df.copy()
                df_mapa[lat_col] = pd.to_numeric(df_mapa[lat_col], errors="coerce")
                df_mapa[lon_col] = pd.to_numeric(df_mapa[lon_col], errors="coerce")
                df_mapa = df_mapa.dropna(subset=[lat_col, lon_col])

                if df_mapa.empty:
                    st.error("Nenhuma coordenada válida foi encontrada na planilha.")
                    st.session_state.df_geo = None
                    st.session_state.mostrar_mapa = False
                else:
                    df_mapa = df_mapa.rename(columns={lat_col: "latitude", lon_col: "longitude"})
                    st.session_state.df_geo = df_mapa.copy()
                    st.session_state.mostrar_mapa = True
                    st.success(f"Mapa preparado com {len(df_mapa)} registros válidos.")
            else:
                st.error("Selecione as colunas de Latitude e Longitude.")

    else:
        endereco_col = st.selectbox("Selecione a coluna com o endereço completo", cols, index=None)
        area_col = st.selectbox("Selecione a coluna de Microárea (opcional)", [None] + cols)
        st.session_state.area_col_atual = area_col

        complemento = st.text_input(
            "Complemento para melhorar a busca (opcional)",
            value="Porto Feliz, São Paulo, Brasil"
        )

        st.caption(f"A conversão vai processar todas as {len(df)} linhas da planilha por padrão.")

        if st.button("Converter endereços"):
            if not endereco_col:
                st.error("Selecione a coluna de endereço.")
            else:
                df_geo = df.copy()
                geolocator = Nominatim(user_agent="mapa_saude_porto_feliz_streamlit")
                geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

                progress = st.progress(0)
                status = st.empty()
                latitudes, longitudes, encontrados_texto = [], [], []
                total = len(df_geo)

                for i, endereco in enumerate(df_geo[endereco_col].fillna("").astype(str)):
                    endereco_busca = endereco.strip()
                    if complemento and endereco_busca:
                        endereco_busca = f"{endereco_busca}, {complemento}"

                    try:
                        status.write(f"Convertendo {i+1}/{total}: {endereco_busca}")
                        location = geocode(endereco_busca)
                        if location:
                            latitudes.append(location.latitude)
                            longitudes.append(location.longitude)
                            encontrados_texto.append(location.address)
                        else:
                            latitudes.append(None)
                            longitudes.append(None)
                            encontrados_texto.append(None)
                    except Exception:
                        latitudes.append(None)
                        longitudes.append(None)
                        encontrados_texto.append(None)

                    progress.progress((i + 1) / total)

                df_geo["latitude"] = latitudes
                df_geo["longitude"] = longitudes
                df_geo["endereco_encontrado"] = encontrados_texto
                st.session_state.df_geo = df_geo.copy()
                st.session_state.mostrar_mapa = False

                encontrados = df_geo["latitude"].notna().sum()
                st.success(f"Conversão concluída. {encontrados} de {total} endereços localizados.")

        if st.session_state.df_geo is not None and st.session_state.modo_atual == "Converter endereço em latitude/longitude":
            df_resultado = st.session_state.df_geo.copy()
            st.write("### Resultado da geocodificação")
            st.dataframe(df_resultado)

            validos = df_resultado.dropna(subset=["latitude", "longitude"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Endereços convertidos", int(validos.shape[0]))
            with col2:
                st.metric("Endereços não localizados", int(df_resultado.shape[0] - validos.shape[0]))

            if not validos.empty:
                if st.button("Gerar mapa com endereços convertidos"):
                    st.session_state.mostrar_mapa = True
            else:
                st.warning("Nenhum endereço foi localizado. Revise o formato dos endereços.")

            csv = df_resultado.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="Baixar planilha com latitude e longitude (CSV)",
                data=csv,
                file_name="enderecos_geocodificados.csv",
                mime="text/csv"
            )

    if st.session_state.mostrar_mapa and st.session_state.df_geo is not None:
        df_mapa_final = st.session_state.df_geo.dropna(subset=["latitude", "longitude"]).copy()

        if df_mapa_final.empty:
            st.warning("Não há coordenadas válidas para exibir no mapa.")
        else:
            centro_lat = df_mapa_final["latitude"].mean()
            centro_lon = df_mapa_final["longitude"].mean()
            mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=13)
            cores = ["red", "blue", "green", "purple", "orange", "darkred"]
            area_col = st.session_state.area_col_atual

            for _, row in df_mapa_final.iterrows():
                cor = "gray"
                if area_col and area_col in df_mapa_final.columns:
                    idx = hash(str(row[area_col])) % len(cores)
                    cor = cores[idx]

                folium.Marker(
                    location=[row["latitude"], row["longitude"]],
                    popup=f"Paciente: {row.get('Paciente', 'N/A')}",
                    tooltip=f"Microárea: {row.get(area_col, 'N/A') if area_col else 'N/A'}",
                    icon=folium.Icon(color=cor)
                ).add_to(mapa)

            st.write("### Mapa")
            st_folium(mapa, width=1000, height=600, returned_objects=[])
else:
    st.info("Aguardando upload da planilha...")