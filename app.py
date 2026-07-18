import io
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Saúde 360 APS", page_icon="📊", layout="wide", initial_sidebar_state="expanded")


def strip_accents(text: str) -> str:
    text = str(text)
    return ''.join(ch for ch in unicodedata.normalize('NFKD', text) if not unicodedata.combining(ch))


def normalize_col(name: str) -> str:
    name = strip_accents(str(name).strip().lower())
    name = re.sub(r"[^a-z0-9]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    vals = series.astype(str).str.strip().str.lower()
    return vals.isin(["1", "true", "sim", "s", "x", "ok", "yes"])


def parse_count(series: pd.Series) -> pd.Series:
    vals = series.astype(str).str.strip().str.lower()
    vals = vals.replace({"": np.nan, "nan": np.nan, "none": np.nan, "n/a": np.nan})
    vals = vals.str.replace("+", "", regex=False)
    return pd.to_numeric(vals, errors="coerce")


def ensure_columns(df: pd.DataFrame, columns: List[str], default=False) -> pd.DataFrame:
    for c in columns:
        if c not in df.columns:
            df[c] = default
    return df


def classificar_score(score: float) -> str:
    if score >= 75:
        return "Ótimo"
    if score >= 50:
        return "Bom"
    if score >= 25:
        return "Suficiente"
    return "Regular"


def faixa_etaria(idade: float) -> str:
    if pd.isna(idade):
        return "Sem idade"
    idade = int(idade)
    if idade < 1:
        return "<1"
    if idade <= 4:
        return "1-4"
    if idade <= 9:
        return "5-9"
    if idade <= 14:
        return "10-14"
    if idade <= 19:
        return "15-19"
    if idade <= 39:
        return "20-39"
    if idade <= 59:
        return "40-59"
    return "60+"


BOA_PRATICA_LABELS = {
    "C4": {
        "consulta_ok": "Realizar ao menos uma consulta semestral, presencial ou remota, para avaliação de risco, adesão ao tratamento e renovação de prescrições",
        "pa_ok": "Registrar ao menos uma aferição de pressão arterial nos últimos 6 meses",
        "antropometria_ok": "Registrar peso e altura para avaliação antropométrica nos últimos 12 meses",
        "visita_ok": "Realizar ao menos duas visitas domiciliares por ACS/TACS, com intervalo mínimo de 30 dias, nos últimos 12 meses",
        "hba1c_ok": "Registrar solicitação ou avaliação de hemoglobina glicada nos últimos 12 meses",
        "pes_ok": "Realizar avaliação dos pés nos últimos 12 meses",
    },
    "C5": {
        "consulta_ok": "Realizar ao menos uma consulta semestral, presencial ou remota, para acompanhamento da pessoa com hipertensão",
        "pa_ok": "Registrar ao menos uma aferição de pressão arterial nos últimos 6 meses",
        "antropometria_ok": "Registrar peso e altura para avaliação antropométrica nos últimos 12 meses",
        "visita_ok": "Realizar ao menos duas visitas domiciliares por ACS/TACS, com intervalo mínimo de 30 dias, nos últimos 12 meses",
    },
    "C6": {
        "consulta_ok": "Registrar ao menos uma consulta presencial ou remota por médico ou enfermeiro nos últimos 12 meses",
        "antropometria_ok": "Registrar peso e altura no mesmo dia para avaliação antropométrica nos últimos 12 meses",
        "visitas_ok": "Realizar ao menos duas visitas domiciliares por ACS/TACS, com intervalo mínimo de 30 dias, nos últimos 12 meses",
        "influenza_ok": "Registrar uma dose da vacina contra influenza nos últimos 12 meses",
    },
}


def label_boa_pratica(indicator_code: str, col: str) -> str:
    return BOA_PRATICA_LABELS.get(indicator_code, {}).get(col, col.replace("_", " ").capitalize())


@dataclass
class IndicatorSpec:
    code: str
    name: str
    type: str
    description: str
    weights: Optional[Dict[str, int]] = None
    non_conditionals: Optional[Dict[str, Callable[[pd.DataFrame], pd.Series]]] = None
    applicability: Optional[Dict[str, str]] = None
    numerator_col: Optional[str] = None
    denominator_col: Optional[str] = None
    entity_label: str = "pessoas"


INDICATORS: Dict[str, IndicatorSpec] = {
    "C4": IndicatorSpec("C4", "Cuidado da pessoa com diabetes", "score", "Pontuação por pessoa com diabetes até 100 pontos.", weights={"consulta_ok": 20, "hba1c_ok": 15, "pes_ok": 15, "visita_ok": 20, "pa_ok": 15, "antropometria_ok": 15}, non_conditionals={"visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)}, entity_label="pessoas com diabetes"),
    "C5": IndicatorSpec("C5", "Cuidado da pessoa com hipertensão", "score", "Pontuação por pessoa com hipertensão até 100 pontos.", weights={"consulta_ok": 25, "pa_ok": 25, "antropometria_ok": 25, "visita_ok": 25}, non_conditionals={"visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)}, entity_label="pessoas com hipertensão"),
    "C6": IndicatorSpec("C6", "Cuidado da pessoa idosa", "score", "Pontuação por pessoa idosa até 100 pontos.", weights={"consulta_ok": 25, "antropometria_ok": 25, "visitas_ok": 25, "influenza_ok": 25}, non_conditionals={"visitas_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)}, entity_label="pessoas idosas"),
}


def calculate_score_indicator(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    df = df.copy()
    weights = spec.weights or {}
    non_conditionals = spec.non_conditionals or {}
    for c in list(weights.keys()):
        if c not in df.columns:
            df[c] = False
    total_score = np.zeros(len(df), dtype=float)
    total_pendencias = np.zeros(len(df), dtype=int)
    for col, weight in weights.items():
        pratica_ok = to_bool(df[col])
        aplicavel = pd.Series(True, index=df.index)
        if col in non_conditionals:
            aplicavel &= ~non_conditionals[col](df).fillna(False).astype(bool)
        total_score += np.where(aplicavel & pratica_ok, weight, 0)
        total_pendencias += np.where(aplicavel & ~pratica_ok, 1, 0)
    df["score"] = total_score
    df["pendencias"] = total_pendencias
    df["classificacao"] = df["score"].apply(classificar_score)
    return df


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = uploaded_file.name.lower()
    if suffix.endswith(".csv"):
        for enc in ["utf-8", "latin1", "cp1252"]:
            uploaded_file.seek(0)
            try:
                return pd.read_csv(uploaded_file, encoding=enc, dtype=str)
            except Exception:
                pass
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, dtype=str)
    uploaded_file.seek(0)
    return pd.read_excel(uploaded_file, dtype=str)


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]
    if "idade" in df.columns:
        df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
    else:
        df["idade"] = np.nan
    for c in ["nome", "equipe", "unidade", "tipo_equipe", "cpf", "cns", "micro_area"]:
        if c not in df.columns:
            df[c] = ""
    if "consulta_medica_enfermagem" in df.columns:
        df["consulta_ok"] = to_bool(df["consulta_medica_enfermagem"])
    if "hemoglobina_glicada" in df.columns:
        df["hba1c_ok"] = to_bool(df["hemoglobina_glicada"])
    if "avaliacao_dos_pes" in df.columns:
        df["pes_ok"] = to_bool(df["avaliacao_dos_pes"])
    if "afericao_de_pa" in df.columns:
        df["pa_ok"] = to_bool(df["afericao_de_pa"])
    if "qtd_registros_de_peso_altura" in df.columns:
        qtd = parse_count(df["qtd_registros_de_peso_altura"])
        df["antropometria_ok"] = qtd.fillna(0).ge(1)
    if "qtd_visitas_domiciliares" in df.columns:
        qtd_vis = parse_count(df["qtd_visitas_domiciliares"])
        df["visita_ok"] = qtd_vis.fillna(0).ge(2)
        df["visitas_ok"] = qtd_vis.fillna(0).ge(2)
    if "vacina_influenza" in df.columns:
        df["influenza_ok"] = to_bool(df["vacina_influenza"])
    df["faixa_etaria"] = df["idade"].apply(faixa_etaria)
    return df


def build_good_practices_df(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    rows = []
    weights = spec.weights or {}
    non_conditionals = spec.non_conditionals or {}
    for col, peso in weights.items():
        if col not in df.columns:
            continue
        aplicavel = pd.Series(True, index=df.index)
        if col in non_conditionals:
            aplicavel &= ~non_conditionals[col](df).fillna(False).astype(bool)
        total_aplicavel = int(aplicavel.sum())
        realizados = int((aplicavel & to_bool(df[col])).sum())
        nao_realizados = max(total_aplicavel - realizados, 0)
        perc = round((realizados / total_aplicavel) * 100, 1) if total_aplicavel else 0.0
        rows.append({
            "Boa prática": label_boa_pratica(spec.code, col),
            "Peso": peso,
            "Realizados": realizados,
            "% Realizado": perc,
            "Não realizado": nao_realizados,
        })
    return pd.DataFrame(rows)


def render_good_practices(df: pd.DataFrame, spec: IndicatorSpec):
    bp_df = build_good_practices_df(df, spec)
    st.markdown("### Cumprimento das boas práticas")
    st.markdown(
        """
        <style>
        .bp-card {
            background: linear-gradient(180deg, #f7fbfb 0%, #eef7f7 100%);
            border: 1px solid #d9ecec;
            border-radius: 18px;
            padding: 18px 18px 10px 18px;
            box-shadow: 0 8px 26px rgba(1, 105, 111, 0.08);
            margin-bottom: 12px;
        }
        .bp-title {font-size: 1.12rem; font-weight: 700; color: #0f3638; margin-bottom: 6px;}
        .bp-sub {font-size: 0.93rem; color: #476466; margin-bottom: 14px;}
        </style>
        <div class="bp-card">
            <div class="bp-title">Painel analítico das boas práticas</div>
            <div class="bp-sub">Resumo por prática com peso, quantidade realizada, percentual realizado e quantidade pendente.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    col_tabela, col_grafico = st.columns([1.45, 1])
    bp_view = bp_df.copy()
    bp_view["% Realizado"] = bp_view["% Realizado"].map(lambda x: f"{x:.1f}%")
    col_tabela.dataframe(bp_view, use_container_width=True, hide_index=True)
    fig = px.bar(
        bp_df.sort_values("% Realizado", ascending=True),
        x="% Realizado",
        y="Boa prática",
        orientation="h",
        text="% Realizado",
        color="% Realizado",
        color_continuous_scale=["#cfe8e8", "#4f98a3", "#01696f"],
        title="Percentual realizado por boa prática",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(height=max(380, len(bp_df) * 72), coloraxis_showscale=False, margin=dict(l=10, r=10, t=50, b=10))
    col_grafico.plotly_chart(fig, use_container_width=True)


def main():
    st.title("Saúde 360 APS testesss")
    with st.sidebar:
        indicator_code = st.selectbox("Indicador", list(INDICATORS.keys()), format_func=lambda x: f"{x} — {INDICATORS[x].name}")
        uploaded = st.file_uploader("Envie CSV, XLS ou XLSX", type=["csv", "xls", "xlsx"])
    if not uploaded:
        st.warning("Envie um arquivo para continuar.")
        return
    raw = read_uploaded_file(uploaded)
    processed = preprocess_df(raw)
    spec = INDICATORS[indicator_code]
    processed = calculate_score_indicator(processed, spec)
    st.metric("Total de registros", len(processed))
    render_good_practices(processed, spec)
    st.dataframe(processed, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
