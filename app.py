import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

st.set_page_config(layout="wide", page_title="Mapa de Saúde Territorial")

st.title("📍 Dashboard de Saúde - Rastreamento Territorial")
st.markdown("Envie uma planilha com endereços ou coordenadas para visualizar os pacientes no mapa.")

uploaded_file = st.file_uploader("Escolha sua planilha (Excel ou CSV)", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.write("### Dados carregados")
    st.dataframe(df.head())

    cols = df.columns.tolist()

    modo = st.radio(
        "Como deseja gerar o mapa?",
        ["Usar latitude/longitude já existentes", "Converter endereço em latitude/longitude"]
    )

    if modo == "Usar latitude/longitude já existentes":
        lat_col = st.selectbox("Selecione a coluna de Latitude", cols, index=None)
        lon_col = st.selectbox("Selecione a coluna de Longitude", cols, index=None)
        area_col = st.selectbox("Selecione a coluna de Microárea (opcional)", [None] + cols)

    else:
        endereco_col = st.selectbox("Selecione a coluna com o endereço completo", cols, index=None)
        area_col = st.selectbox("Selecione a coluna de Microárea (opcional)", [None] + cols)

        if st.button("Converter endereços"):
            if endereco_col:
                geolocator = Nominatim(user_agent="mapa_saude_porto_feliz")
                geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

                df["endereco_busca"] = df[endereco_col].astype(str).fillna("")
                df["location"] = df["endereco_busca"].apply(geocode)
                df["latitude"] = df["location"].apply(lambda loc: loc.latitude if loc else None)
                df["longitude"] = df["location"].apply(lambda loc: loc.longitude if loc else None)

                st.success("Conversão concluída.")
                st.dataframe(df[[endereco_col, "latitude", "longitude"]].head())

                lat_col = "latitude"
                lon_col = "longitude"

                mapa = folium.Map(location=[-23.218, -47.520], zoom_start=13)
                cores = ["red", "blue", "green", "purple", "orange", "darkred"]

                for _, row in df.dropna(subset=[lat_col, lon_col]).iterrows():
                    cor = "gray"
                    if area_col:
                        idx = hash(str(row[area_col])) % len(cores)
                        cor = cores[idx]

                    folium.Marker(
                        location=[row[lat_col], row[lon_col]],
                        popup=f"Paciente: {row.get('Paciente', 'N/A')}",
                        tooltip=f"Microárea: {row.get(area_col, 'N/A') if area_col else 'N/A'}",
                        icon=folium.Icon(color=cor)
                    ).add_to(mapa)

                st_folium(mapa, width=1000, height=600)
            else:
                st.error("Selecione a coluna de endereço.")
