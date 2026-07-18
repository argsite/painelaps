import io
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Saúde 360 APS",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# Utilidades
# =========================
def strip_accents(text: str) -> str:
    text = str(text)
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


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


def ensure_column(df: pd.DataFrame, col: str, default=None):
    if col not in df.columns:
        df[col] = default


def first_existing(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def map_first(df: pd.DataFrame, target: str, candidates: List[str], default=""):
    src = first_existing(df, candidates)
    if src and target not in df.columns:
        df[target] = df[src]
    elif target not in df.columns:
        df[target] = default


def infer_tipo_equipe_from_text(series: pd.Series) -> pd.Series:
    vals = series.astype(str).str.upper()
    out = np.where(vals.str.contains(" 76") | vals.str.contains("TIPO 76"), "76", "")
    out = np.where((pd.Series(out, index=series.index) == "") & (vals.str.contains(" 70") | vals.str.contains("TIPO 70")), "70", out)
    return pd.Series(out, index=series.index)


# =========================
# Especificações dos indicadores
# =========================
@dataclass
class IndicatorSpec:
    code: str
    name: str
    type: str  # score | percentual
    description: str
    weights: Dict[str, int] = field(default_factory=dict)
    non_conditionals: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(default_factory=dict)
    numerator_col: Optional[str] = None
    denominator_col: Optional[str] = None
    entity_label: str = "pessoas"
    applicable_age_rule: Optional[Callable[[pd.DataFrame], pd.Series]] = None


BOA_PRATICA_LABELS = {
    "C1": {
        "cadastro_ok": "Cadastro individual atualizado e vínculo ativo no território",
        "atendimento_ok": "Pessoa acompanhada/atendida conforme relatório operacional",
    },
    "C2": {
        "consulta_ok": "Consulta da criança registrada no período",
        "vacina_ok": "Vacinação/aplicação preventiva registrada no período",
        "antropometria_ok": "Peso e altura registrados para acompanhamento do desenvolvimento",
        "visita_ok": "Visita domiciliar/acompanhamento territorial registrado",
    },
    "C3": {
        "consulta_ok": "Consulta de pré-natal/puerpério registrada no período",
        "pa_ok": "Aferição de pressão arterial registrada",
        "antropometria_ok": "Peso e/ou avaliação antropométrica registrada",
        "visita_ok": "Visitas domiciliares registradas",
        "exame_ok": "Exame/solicitação importante registrado no período",
    },
    "C4": {
        "consulta_ok": "Realizar ao menos uma consulta semestral, presencial ou remota",
        "pa_ok": "Registrar ao menos uma aferição de pressão arterial nos últimos 6 meses",
        "antropometria_ok": "Registrar peso e altura para avaliação antropométrica nos últimos 12 meses",
        "visita_ok": "Realizar ao menos duas visitas domiciliares por ACS/TACS, nos últimos 12 meses",
        "hba1c_ok": "Registrar solicitação ou avaliação de hemoglobina glicada nos últimos 12 meses",
        "pes_ok": "Realizar avaliação dos pés nos últimos 12 meses",
    },
    "C5": {
        "consulta_ok": "Realizar ao menos uma consulta semestral, presencial ou remota",
        "pa_ok": "Registrar ao menos uma aferição de pressão arterial nos últimos 6 meses",
        "antropometria_ok": "Registrar peso e altura para avaliação antropométrica nos últimos 12 meses",
        "visita_ok": "Realizar ao menos duas visitas domiciliares por ACS/TACS, nos últimos 12 meses",
    },
    "C6": {
        "consulta_ok": "Registrar consulta presencial ou remota por médico ou enfermeiro",
        "antropometria_ok": "Registrar peso e altura no mesmo dia para avaliação antropométrica",
        "visitas_ok": "Realizar ao menos duas visitas domiciliares por ACS/TACS",
        "influenza_ok": "Registrar uma dose da vacina contra influenza nos últimos 12 meses",
    },
    "C7": {
        "citopatologico_ok": "Citopatológico/ação preventiva registrado",
        "mamografia_ok": "Mamografia/ação correlata registrada quando aplicável",
        "consulta_ok": "Consulta de cuidado/prevenção registrada no período",
        "visita_ok": "Acompanhamento domiciliar/territorial registrado",
    },
}


def label_boa_pratica(indicator_code: str, col: str) -> str:
    return BOA_PRATICA_LABELS.get(indicator_code, {}).get(col, col.replace("_", " ").capitalize())


INDICATORS: Dict[str, IndicatorSpec] = {
    "C1": IndicatorSpec(
        code="C1",
        name="Mais acesso",
        type="percentual",
        description="Indicador operacional local de acesso/vínculo a partir do relatório importado.",
        numerator_col="numerador_c1",
        denominator_col="denominador_c1",
        entity_label="pessoas cadastradas",
    ),
    "C2": IndicatorSpec(
        code="C2",
        name="Cuidado no desenvolvimento infantil",
        type="score",
        description="Painel operacional local para acompanhamento do desenvolvimento infantil.",
        weights={"consulta_ok": 30, "vacina_ok": 25, "antropometria_ok": 25, "visita_ok": 20},
        non_conditionals={},
        entity_label="crianças acompanhadas",
    ),
    "C3": IndicatorSpec(
        code="C3",
        name="Cuidado na gestação e puerpério",
        type="score",
        description="Painel operacional local para gestantes e puérperas.",
        weights={"consulta_ok": 30, "pa_ok": 20, "antropometria_ok": 15, "visita_ok": 15, "exame_ok": 20},
        non_conditionals={},
        entity_label="gestantes/puérperas",
    ),
    "C4": IndicatorSpec(
        code="C4",
        name="Cuidado da pessoa com diabetes",
        type="score",
        description="Pontuação por pessoa com diabetes até 100 pontos.",
        weights={"consulta_ok": 20, "hba1c_ok": 15, "pes_ok": 15, "visita_ok": 20, "pa_ok": 15, "antropometria_ok": 15},
        non_conditionals={"visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)},
        entity_label="pessoas com diabetes",
    ),
    "C5": IndicatorSpec(
        code="C5",
        name="Cuidado da pessoa com hipertensão",
        type="score",
        description="Pontuação por pessoa com hipertensão até 100 pontos.",
        weights={"consulta_ok": 25, "pa_ok": 25, "antropometria_ok": 25, "visita_ok": 25},
        non_conditionals={"visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)},
        entity_label="pessoas com hipertensão",
    ),
    "C6": IndicatorSpec(
        code="C6",
        name="Cuidado da pessoa idosa",
        type="score",
        description="Pontuação por pessoa idosa até 100 pontos.",
        weights={"consulta_ok": 25, "antropometria_ok": 25, "visitas_ok": 25, "influenza_ok": 25},
        non_conditionals={"visitas_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)},
        entity_label="pessoas idosas",
    ),
    "C7": IndicatorSpec(
        code="C7",
        name="Cuidado da mulher na prevenção do câncer",
        type="score",
        description="Painel operacional local para prevenção do câncer da mulher.",
        weights={"citopatologico_ok": 35, "mamografia_ok": 25, "consulta_ok": 20, "visita_ok": 20},
        non_conditionals={},
        entity_label="mulheres acompanhadas",
    ),
}


# =========================
# Leitura e identificação
# =========================
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


def detect_indicator_from_columns(df: pd.DataFrame, filename: str) -> Optional[str]:
    cols = set(df.columns)
    name = normalize_col(filename)

    if {"hemoglobina_glicada", "avaliacao_dos_pes"}.issubset(cols) or "diabetes" in name:
        return "C4"
    if "hipertensao" in name or {"afericao_de_pa", "qtd_registros_de_peso_altura"}.issubset(cols):
        if "hemoglobina_glicada" not in cols and "avaliacao_dos_pes" not in cols:
            return "C5"
    if "idosa" in name or "idoso" in name or "vacina_influenza" in cols:
        return "C6"
    if "gestante" in name or "puerpera" in name or "gestacao" in name:
        return "C3"
    if "desenvolvimento_infantil" in name or "infantil" in name or "crianca" in name:
        return "C2"
    if "cancer" in name or "mulher" in name or "mamografia" in cols or "citopatologico" in cols:
        return "C7"
    if "acesso" in name:
        return "C1"
    return None


# =========================
# Pré-processamento genérico
# =========================
def preprocess_df(df: pd.DataFrame, indicator_code: Optional[str] = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]

    map_first(df, "nome", ["nome", "nome_completo", "cidadao", "usuario", "paciente"])
    map_first(df, "cpf", ["cpf"])
    map_first(df, "cns", ["cns", "cns_cidadao", "cartao_sus"])
    map_first(df, "idade", ["idade"])
    map_first(df, "endereco", ["endereco", "logradouro"])
    map_first(df, "equipe", ["equipe", "equipe_area", "equipe_de_area"])
    map_first(df, "micro_area", ["micro_area", "microarea"])
    map_first(df, "equipe_vinculo", ["equipe_vinculo", "equipe_de_vinculo"])
    map_first(df, "cadastro_atualizado", ["cadastro_atualizado"])
    map_first(df, "data_atualizacao_cadastro", ["data_atualizacao_cadastro"])
    map_first(df, "acompanhado", ["acompanhado"])

    if "idade" in df.columns:
        df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
    else:
        df["idade"] = np.nan
    df["faixa_etaria"] = df["idade"].apply(faixa_etaria)

    if "tipo_equipe" not in df.columns:
        if "equipe_vinculo" in df.columns:
            df["tipo_equipe"] = infer_tipo_equipe_from_text(df["equipe_vinculo"])
        else:
            df["tipo_equipe"] = ""

    # Flags clínicas e operacionais genéricas
    if "consulta_medica_enfermagem" in df.columns:
        df["consulta_ok"] = to_bool(df["consulta_medica_enfermagem"])
    elif "consulta" in df.columns:
        df["consulta_ok"] = to_bool(df["consulta"])
    else:
        df["consulta_ok"] = False

    if "afericao_de_pa" in df.columns:
        df["pa_ok"] = to_bool(df["afericao_de_pa"])
    else:
        df["pa_ok"] = False

    if "hemoglobina_glicada" in df.columns:
        df["hba1c_ok"] = to_bool(df["hemoglobina_glicada"])
    else:
        df["hba1c_ok"] = False

    if "avaliacao_dos_pes" in df.columns:
        df["pes_ok"] = to_bool(df["avaliacao_dos_pes"])
    else:
        df["pes_ok"] = False

    if "qtd_registros_de_peso_altura" in df.columns:
        qtd = parse_count(df["qtd_registros_de_peso_altura"])
        df["antropometria_ok"] = qtd.fillna(0).ge(1)
    elif "peso_altura" in df.columns:
        df["antropometria_ok"] = to_bool(df["peso_altura"])
    else:
        df["antropometria_ok"] = False

    if "qtd_visitas_domiciliares" in df.columns:
        qtd_vis = parse_count(df["qtd_visitas_domiciliares"])
        df["visita_ok"] = qtd_vis.fillna(0).ge(2)
        df["visitas_ok"] = df["visita_ok"]
    elif "visita_domiciliar" in df.columns:
        df["visita_ok"] = to_bool(df["visita_domiciliar"])
        df["visitas_ok"] = df["visita_ok"]
    else:
        df["visita_ok"] = False
        df["visitas_ok"] = False

    if "vacina_influenza" in df.columns:
        df["influenza_ok"] = to_bool(df["vacina_influenza"])
    else:
        df["influenza_ok"] = False

    # C1 - acesso local
    df["cadastro_ok"] = to_bool(df["cadastro_atualizado"]) if "cadastro_atualizado" in df.columns else False
    df["atendimento_ok"] = to_bool(df["acompanhado"]) if "acompanhado" in df.columns else df["consulta_ok"]
    df["numerador_c1"] = (df["cadastro_ok"] | df["atendimento_ok"]).astype(int)
    df["denominador_c1"] = 1

    # C2 - desenvolvimento infantil (operacional local)
    if indicator_code == "C2" or (indicator_code is None and df["idade"].notna().any()):
        df["vacina_ok"] = to_bool(df["vacina_influenza"]) if "vacina_influenza" in df.columns else False
        if "acompanhado" in df.columns:
            df["vacina_ok"] = df["vacina_ok"] | to_bool(df["acompanhado"])

    # C3 - gestação/puerpério (operacional local)
    df["exame_ok"] = False
    possible_exam_cols = [c for c in df.columns if any(k in c for k in ["exame", "teste", "hemoglobina", "citopatologico", "mamografia"])]
    if possible_exam_cols:
        temp = pd.Series(False, index=df.index)
        for c in possible_exam_cols:
            temp = temp | to_bool(df[c])
        df["exame_ok"] = temp

    # C7 - mulher / prevenção do câncer (operacional local)
    df["citopatologico_ok"] = False
    df["mamografia_ok"] = False
    if "citopatologico" in df.columns:
        df["citopatologico_ok"] = to_bool(df["citopatologico"])
    elif "acompanhado" in df.columns and indicator_code == "C7":
        df["citopatologico_ok"] = to_bool(df["acompanhado"])

    if "mamografia" in df.columns:
        df["mamografia_ok"] = to_bool(df["mamografia"])

    return df


# =========================
# Cálculos
# =========================
def calculate_score_indicator(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    df = df.copy()
    weights = spec.weights or {}
    non_conditionals = spec.non_conditionals or {}

    for c in list(weights.keys()):
        ensure_column(df, c, False)

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


def calculate_percentual_indicator(df: pd.DataFrame, spec: IndicatorSpec) -> Tuple[pd.DataFrame, float]:
    df = df.copy()
    num = pd.to_numeric(df[spec.numerator_col], errors="coerce").fillna(0) if spec.numerator_col else pd.Series(0, index=df.index)
    den = pd.to_numeric(df[spec.denominator_col], errors="coerce").fillna(0) if spec.denominator_col else pd.Series(0, index=df.index)
    df["numerador"] = num
    df["denominador"] = den
    total_num = num.sum()
    total_den = den.sum()
    indicador = (total_num / total_den * 100) if total_den > 0 else 0
    df["score"] = np.where(den > 0, (num / den) * 100, 0)
    df["classificacao"] = df["score"].apply(classificar_score)
    df["pendencias"] = np.where(num > 0, 0, 1)
    return df, indicador


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
        rows.append(
            {
                "Boa prática": label_boa_pratica(spec.code, col),
                "Peso": peso,
                "Realizados": realizados,
                "% Realizado": perc,
                "Não realizado": nao_realizados,
            }
        )
    return pd.DataFrame(rows)


# =========================
# Renderização
# =========================
def render_good_practices(df: pd.DataFrame, spec: IndicatorSpec):
    bp_df = build_good_practices_df(df, spec)
    st.markdown("### Cumprimento das boas práticas")
    if bp_df.empty:
        st.info("Não foi possível identificar boas práticas estruturadas para este relatório.")
        return
    st.dataframe(bp_df, use_container_width=True)
    fig = px.bar(
        bp_df,
        x="Boa prática",
        y="% Realizado",
        text="% Realizado",
        title="Percentual de realização por boa prática",
    )
    fig.update_layout(xaxis_title="", yaxis_title="%")
    st.plotly_chart(fig, use_container_width=True)


def export_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="dados")
    buffer.seek(0)
    return buffer.read()


def render_score_dashboard(df: pd.DataFrame, spec: IndicatorSpec):
    df_scored = calculate_score_indicator(df, spec)

    total = len(df_scored)
    media_score = df_scored["score"].mean() if total > 0 else 0
    otimos = int((df_scored["classificacao"] == "Ótimo").sum())
    bons = int((df_scored["classificacao"] == "Bom").sum())
    suficientes = int((df_scored["classificacao"] == "Suficiente").sum())
    regulares = int((df_scored["classificacao"] == "Regular").sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total", total)
    c2.metric("Score médio", f"{media_score:.1f}")
    c3.metric("Ótimo", otimos)
    c4.metric("Bom", bons)
    c5.metric("Pendência alta", regulares + suficientes)

    colg1, colg2 = st.columns(2)
    with colg1:
        if total > 0:
            fig_score = px.histogram(df_scored, x="score", nbins=10, title="Distribuição de score")
            st.plotly_chart(fig_score, use_container_width=True)
    with colg2:
        class_df = df_scored["classificacao"].value_counts().reset_index()
        class_df.columns = ["Classificação", "Quantidade"]
        fig_class = px.pie(class_df, names="Classificação", values="Quantidade", title="Classificação")
        st.plotly_chart(fig_class, use_container_width=True)

    render_good_practices(df_scored, spec)

    st.markdown("### Consolidado por equipe")
    if "equipe" in df_scored.columns:
        by_team = (
            df_scored.groupby("equipe", dropna=False)
            .agg(total=("nome", "count"), score_medio=("score", "mean"), pendencias=("pendencias", "sum"))
            .reset_index()
            .sort_values(["score_medio", "total"], ascending=[False, False])
        )
        st.dataframe(by_team, use_container_width=True)

    render_nominal(df_scored, spec)


def render_percentual_dashboard(df: pd.DataFrame, spec: IndicatorSpec):
    df_calc, indicador = calculate_percentual_indicator(df, spec)
    total = len(df_calc)
    cobertos = int(pd.to_numeric(df_calc["numerador"], errors="coerce").fillna(0).sum())
    elegiveis = int(pd.to_numeric(df_calc["denominador"], errors="coerce").fillna(0).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de registros", total)
    c2.metric("Numerador", cobertos)
    c3.metric("Indicador", f"{indicador:.1f}%")

    if "equipe" in df_calc.columns:
        by_team = (
            df_calc.groupby("equipe", dropna=False)
            .agg(numerador=("numerador", "sum"), denominador=("denominador", "sum"))
            .reset_index()
        )
        by_team["percentual"] = np.where(by_team["denominador"] > 0, by_team["numerador"] / by_team["denominador"] * 100, 0)
        st.dataframe(by_team, use_container_width=True)
        fig = px.bar(by_team, x="equipe", y="percentual", title="Indicador por equipe")
        st.plotly_chart(fig, use_container_width=True)

    render_nominal(df_calc, spec)


def render_nominal(df: pd.DataFrame, spec: IndicatorSpec):
    st.markdown("### Lista nominal")
    equipes = sorted([str(e) for e in df.get("equipe", pd.Series(dtype=str)).dropna().unique() if str(e).strip()])
    microareas = sorted([str(m) for m in df.get("micro_area", pd.Series(dtype=str)).dropna().unique() if str(m).strip()])

    f1, f2, f3 = st.columns(3)
    equipe_sel = f1.selectbox("Equipe", ["(todas)"] + equipes, key=f"eq_{spec.code}")
    micro_sel = f2.selectbox("Microárea", ["(todas)"] + microareas, key=f"ma_{spec.code}")
    classe_sel = f3.selectbox("Classificação", ["(todas)", "Ótimo", "Bom", "Suficiente", "Regular"], key=f"cl_{spec.code}")

    view = df.copy()
    if equipe_sel != "(todas)":
        view = view[view["equipe"].astype(str) == equipe_sel]
    if micro_sel != "(todas)":
        view = view[view["micro_area"].astype(str) == micro_sel]
    if classe_sel != "(todas)" and "classificacao" in view.columns:
        view = view[view["classificacao"] == classe_sel]

    preferred_cols = [
        "nome", "cpf", "cns", "idade", "faixa_etaria", "endereco",
        "equipe", "micro_area", "equipe_vinculo", "tipo_equipe",
        "score", "classificacao", "pendencias",
        "cadastro_ok", "atendimento_ok", "consulta_ok", "pa_ok",
        "antropometria_ok", "visita_ok", "visitas_ok", "hba1c_ok",
        "pes_ok", "influenza_ok", "citopatologico_ok", "mamografia_ok", "exame_ok",
    ]
    cols = [c for c in preferred_cols if c in view.columns]
    if not cols:
        cols = list(view.columns)

    st.dataframe(view[cols], use_container_width=True, height=420)

    csv_bytes = view[cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Baixar CSV filtrado",
        data=csv_bytes,
        file_name=f"{spec.code.lower()}_lista_filtrada.csv",
        mime="text/csv",
    )
    st.download_button(
        "Baixar Excel filtrado",
        data=export_excel_bytes(view[cols]),
        file_name=f"{spec.code.lower()}_lista_filtrada.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# =========================
# Aplicação
# =========================
def main():
    st.title("Saúde 360 APS - Dashboard multipainel")
    st.caption("Painel expandido para os 7 indicadores com leitura flexível de relatórios e cálculo operacional local.")

    st.sidebar.header("Importação")
    uploaded_file = st.sidebar.file_uploader("Envie um relatório CSV/XLS/XLSX", type=["csv", "xls", "xlsx"])

    st.sidebar.header("Indicador")
    manual_indicator = st.sidebar.selectbox(
        "Selecionar manualmente (opcional)",
        ["Automático"] + [f"{k} - {v.name}" for k, v in INDICATORS.items()],
    )

    if uploaded_file is None:
        st.info("Envie um relatório para começar. O app tenta identificar o indicador automaticamente pelo nome do arquivo e pelas colunas.")
        st.stop()

    try:
        df_raw = read_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    detected = detect_indicator_from_columns(pd.DataFrame(columns=[normalize_col(c) for c in df_raw.columns]), uploaded_file.name)
    selected_code = manual_indicator.split(" ")[0] if manual_indicator != "Automático" else detected
    if selected_code is None:
        st.warning("Não foi possível identificar automaticamente o indicador. Escolha manualmente na barra lateral.")
        st.stop()

    spec = INDICATORS[selected_code]
    df = preprocess_df(df_raw, selected_code)

    st.success(f"Indicador em análise: {spec.code} - {spec.name}")

    tab1, tab2, tab3 = st.tabs(["Painel", "Qualidade dos dados", "Prévia do arquivo"])

    with tab1:
        st.markdown(f"## {spec.code} - {spec.name}")
        st.write(spec.description)
        if spec.type == "score":
            render_score_dashboard(df, spec)
        else:
            render_percentual_dashboard(df, spec)

    with tab2:
        st.markdown("## Qualidade dos dados")
        quality = pd.DataFrame(
            {
                "coluna": df.columns,
                "nulos": [int(df[c].isna().sum()) for c in df.columns],
                "vazios": [int((df[c].astype(str).str.strip() == "").sum()) for c in df.columns],
                "unicos": [int(df[c].nunique(dropna=True)) for c in df.columns],
            }
        )
        st.dataframe(quality, use_container_width=True)
        st.download_button(
            "Baixar relatório de qualidade (CSV)",
            data=quality.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"qualidade_{spec.code.lower()}.csv",
            mime="text/csv",
        )

    with tab3:
        st.markdown("## Prévia do arquivo importado")
        st.dataframe(df.head(200), use_container_width=True, height=520)


if __name__ == "__main__":
    main()
