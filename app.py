import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(layout="wide", page_title="Mapa de Saúde Territorial")

if "df_geo" not in st.session_state:
    st.session_state.df_geo = None

st.title("📍 Dashboard de Saúde - Rastreamento Territorial")

uploaded_file = st.file_uploader("Escolha sua planilha (Excel ou CSV)", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    cols = df.columns.tolist()
    endereco_col = st.selectbox("Selecione a coluna com o endereço completo", cols, index=None)
    area_col = st.selectbox("Selecione a coluna de Microárea (opcional)", [None] + cols)

    if st.button("Converter endereços"):
        if endereco_col:
            geolocator = Nominatim(user_agent="mapa_saude_porto_feliz")
            geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

            progress = st.progress(0)
            status = st.empty()

            latitudes = []
            longitudes = []

            total = len(df)

            for i, endereco in enumerate(df[endereco_col].fillna("").astype(str)):
                try:
                    status.write(f"Convertendo {i+1}/{total}: {endereco}")
                    location = geocode(endereco)
                    if location:
                        latitudes.append(location.latitude)
                        longitudes.append(location.longitude)
                    else:
                        latitudes.append(None)
                        longitudes.append(None)
                except Exception:
                    latitudes.append(None)
                    longitudes.append(None)

                progress.progress((i + 1) / total)

            df["latitude"] = latitudes
            df["longitude"] = longitudes
            st.session_state.df_geo = df.copy()

            encontrados = df["latitude"].notna().sum()
            st.success(f"Conversão concluída. {encontrados} de {total} endereços localizados.")

    if st.session_state.df_geo is not None:
        df_mapa = st.session_state.df_geo

        st.write("### Resultado da geocodificação")
        st.dataframe(df_mapa.head())

        if st.button("Gerar mapa"):
            validos = df_mapa.dropna(subset=["latitude", "longitude"])

            if not validos.empty:
                centro_lat = validos["latitude"].mean()
                centro_lon = validos["longitude"].mean()

                mapa = folium.Map(location=[centro_lat, centro_lon], zoom_start=13)
                cores = ["red", "blue", "green", "purple", "orange", "darkred"]

                for _, row in validos.iterrows():
                    cor = "gray"
                    if area_col:
                        idx = hash(str(row[area_col])) % len(cores)
                        cor = cores[idx]

                    folium.Marker(
                        location=[row["latitude"], row["longitude"]],
                        popup=f"Paciente: {row.get('Paciente', 'N/A')}",
                        tooltip=f"Microárea: {row.get(area_col, 'N/A') if area_col else 'N/A'}",
                        icon=folium.Icon(color=cor)
                    ).add_to(mapa)

                st_folium(mapa, width=1000, height=600)
            else:
                st.error("Nenhum endereço foi convertido com sucesso.")
