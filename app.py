import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import geopandas as gpd

# Configuração da página
st.set_page_config(layout="wide", page_title="Mapa de Saúde Territorial")

st.title("📍 Dashboard de Saúde - Rastreamento Territorial")
st.markdown("Faça o upload da sua planilha para visualizar os pacientes no mapa.")

# 1. Upload do arquivo
uploaded_file = st.file_uploader("Escolha sua planilha (Excel ou CSV)", type=["xlsx", "csv"])

if uploaded_file:
    # Carregar dados
    if uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
    
    st.write("### Dados Carregados:")
    st.dataframe(df.head())

    # Seleção de colunas (para o usuário mapear o que é o quê)
    cols = df.columns.tolist()
    lat_col = st.selectbox("Selecione a coluna de Latitude", cols, index=None)
    lon_col = st.selectbox("Selecione a coluna de Longitude", cols, index=None)
    area_col = st.selectbox("Selecione a coluna de Microárea (para cores)", cols, index=None)

    if st.button("Gerar Mapa"):
        if lat_col and lon_col:
            # Centro do mapa (Porto Feliz)
            mapa = folium.Map(location=[-23.218, -47.520], zoom_start=14)
            
            # Cores dinâmicas para microáreas
            cores = ['red', 'blue', 'green', 'purple', 'orange', 'darkred']
            
            # Adicionar marcadores
            for i, row in df.iterrows():
                cor = 'gray'
                if area_col:
                    # Atribui cor baseada na microárea
                    idx = hash(str(row[area_col])) % len(cores)
                    cor = cores[idx]
                
                folium.Marker(
                    location=[row[lat_col], row[lon_col]],
                    popup=f"Paciente: {row.get('Paciente', 'N/A')}",
                    tooltip=f"Microárea: {row.get(area_col, 'N/A')}",
                    icon=folium.Icon(color=cor)
                ).add_to(mapa)
            
            st.success("Mapa gerado com sucesso!")
            st_folium(mapa, width=1000, height=600)
        else:
            st.error("Por favor, selecione as colunas de Latitude e Longitude.")
else:
    st.info("Aguardando upload da planilha...")
```

### Notas Importantes para o Sucesso:

1.  **Latitude e Longitude:** Como discutimos, o Folium precisa de coordenadas numéricas. Certifique-se de que sua planilha tenha colunas com valores como `-23.218` e `-47.520`. Se a sua planilha tiver apenas o endereço escrito, o próximo passo será integrar uma função de geocodificação (como a biblioteca `geopy`) para converter esses endereços antes de gerar o mapa.
2.  **Polígonos (Áreas das Equipes):** Para adicionar as suas camadas do Google Maps (GeoJSON), basta adicionar isto logo após a criação do `mapa = folium.Map(...)`:
    ```python
    # Exemplo de como carregar suas áreas desenhadas
    # folium.GeoJson("suas_areas.geojson").add_to(mapa)