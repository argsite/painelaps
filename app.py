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

# ---------------------
# Utilitários básicos
# ---------------------

def strip_accents(text: str) -> str:
    text = str(text)
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


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


# ---------------------
# Especificação de indicadores
# ---------------------

@dataclass
class IndicatorSpec:
    code: str
    name: str
    type: str  # "score" ou "percentual"
    description: str
    weights: Optional[Dict[str, int]] = None
    non_conditionals: Optional[Dict[str, Callable[[pd.DataFrame], pd.Series]]] = None
    applicability: Optional[Dict[str, str]] = None
    numerator_col: Optional[str] = None
    denominator_col: Optional[str] = None
    entity_label: str = "pessoas"


INDICATORS: Dict[str, IndicatorSpec] = {
    "C1": IndicatorSpec(
        "C1",
        "Mais acesso na APS",
        "percentual",
        "Percentual de atendimentos programados em relação ao total de atendimentos válidos.",
        numerator_col="demanda_programada",
        denominator_col="atendimento_valido",
        entity_label="atendimentos",
    ),
    "C2": IndicatorSpec(
        "C2",
        "Cuidado no desenvolvimento infantil",
        "score",
        "Monitoramento da criança com base em boas práticas registradas.",
        weights={
            "consulta_ok": 20,
            "vacina_ok": 20,
            "peso_altura_ok": 20,
            "visita_ok": 20,
            "desenvolvimento_ok": 20,
        },
        non_conditionals={
            "visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index),
        },
        entity_label="crianças",
    ),
    "C3": IndicatorSpec(
        "C3",
        "Cuidado na gestação e puerpério",
        "score",
        "Pontuação por gestante/puérpera até 100 pontos.",
        weights={
            "pre_natal_12s_ok": 10,
            "consultas_gest_ok": 9,
            "pa_ok": 9,
            "antropometria_ok": 9,
            "visitas_gest_ok": 9,
            "dtpa_ok": 9,
            "tri1_ok": 9,
            "tri3_ok": 9,
            "puerperio_consulta_ok": 9,
            "puerperio_visita_ok": 9,
            "odonto_ok": 9,
        },
        non_conditionals={
            "visitas_gest_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index),
            "puerperio_visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index),
        },
        entity_label="gestantes/puérperas",
    ),
    "C4": IndicatorSpec(
        "C4",
        "Cuidado da pessoa com diabetes",
        "score",
        "Pontuação por pessoa com diabetes até 100 pontos.",
        weights={
            "consulta_ok": 20,
            "hba1c_ok": 20,
            "solicitacao_ok": 15,
            "pes_ok": 15,
            "retina_ok": 15,
            "visita_ok": 15,
        },
        non_conditionals={
            "visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index),
        },
        entity_label="pessoas com diabetes",
    ),
    "C5": IndicatorSpec(
        "C5",
        "Cuidado da pessoa com hipertensão",
        "score",
        "Pontuação por pessoa com hipertensão até 100 pontos.",
        weights={
            "consulta_ok": 25,
            "pa_ok": 25,
            "antropometria_ok": 25,
            "visita_ok": 25,
        },
        non_conditionals={
            "visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index),
        },
        entity_label="pessoas com hipertensão",
    ),
    "C6": IndicatorSpec(
        "C6",
        "Cuidado da pessoa idosa",
        "score",
        "Pontuação por pessoa idosa até 100 pontos.",
        weights={
            "consulta_ok": 25,
            "antropometria_ok": 25,
            "visitas_ok": 25,
            "influenza_ok": 25,
        },
        non_conditionals={
            "visitas_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index),
        },
        entity_label="pessoas idosas",
    ),
    "C7": IndicatorSpec(
        "C7",
        "Cuidado da mulher na prevenção do câncer",
        "score",
        "Pontuação por mulher elegível conforme faixa etária e práticas aplicáveis.",
        weights={
            "colo_utero_ok": 20,
            "hpv_ok": 30,
            "saude_reprodutiva_ok": 30,
            "mama_ok": 20,
        },
        applicability={
            "colo_utero_ok": "colo_utero_aplicavel",
            "hpv_ok": "hpv_aplicavel",
            "saude_reprodutiva_ok": "saude_reprodutiva_aplicavel",
            "mama_ok": "mama_aplicavel",
        },
        entity_label="mulheres",
    ),
}

# Descrições de boas práticas (parciais, ajustáveis)
BOA_PRATICA_LABELS = {
    "C4": {
        "consulta_ok": "Consulta semestral para avaliação de risco, adesão e prescrição",
        "hba1c_ok": "Solicitação ou avaliação de hemoglobina glicada em 12 meses",
        "pes_ok": "Avaliação dos pés em 12 meses",
        "visita_ok": "Duas visitas domiciliares em 12 meses (ACS/TACS)",
        "pa_ok": "Aferição de pressão arterial em 6 meses",
        "antropometria_ok": "Registro de peso e altura em 12 meses",
    },
    "C5": {
        "consulta_ok": "Consulta semestral para acompanhamento da pessoa com hipertensão",
        "pa_ok": "Aferição de pressão arterial em 6 meses",
        "antropometria_ok": "Peso e altura registrados em 12 meses",
        "visita_ok": "Duas visitas domiciliares em 12 meses (ACS/TACS)",
    },
    "C6": {
        "consulta_ok": "Consulta em 12 meses por médico ou enfermeiro",
        "antropometria_ok": "Peso e altura registrados no mesmo dia em 12 meses",
        "visitas_ok": "Duas visitas domiciliares em 12 meses (ACS/TACS)",
        "influenza_ok": "Dose de vacina influenza em 12 meses",
    },
}


def label_boa_pratica(indicator_code: str, col: str) -> str:
    return BOA_PRATICA_LABELS.get(indicator_code, {}).get(col, col.replace("_", " ").capitalize())


# ---------------------
# Cálculo de indicadores
# ---------------------

def calculate_score_indicator(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    df = df.copy()
    weights = spec.weights or {}
    non_conditionals = spec.non_conditionals or {}
    applicability = spec.applicability or {}

    for c in list(weights.keys()) + list(applicability.values()):
        if c not in df.columns:
            df[c] = False

    total_score = np.zeros(len(df), dtype=float)
    total_pendencias = np.zeros(len(df), dtype=int)
    total_aplicaveis = np.zeros(len(df), dtype=int)

    for col, weight in weights.items():
        pratica_ok = to_bool(df[col])
        aplicavel = pd.Series(True, index=df.index)
        if col in applicability:
            aplicavel &= to_bool(df[applicability[col]])
        if col in non_conditionals:
            aplicavel &= ~non_conditionals[col](df).fillna(False).astype(bool)

        total_score += np.where(aplicavel & pratica_ok, weight, 0)
        total_pendencias += np.where(aplicavel & ~pratica_ok, 1, 0)
        total_aplicaveis += np.where(aplicavel, 1, 0)

    df["score"] = total_score
    df["pendencias"] = total_pendencias
    df["praticas_aplicaveis"] = total_aplicaveis
    df["classificacao"] = df["score"].apply(classificar_score)
    return df


def calculate_percent_indicator(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    df = df.copy()
    ensure_columns(df, [spec.numerator_col, spec.denominator_col], default=False)
    df[spec.numerator_col] = to_bool(df[spec.numerator_col])
    df[spec.denominator_col] = to_bool(df[spec.denominator_col])
    return df


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    suffix = uploaded_file.name.lower()
    if suffix.endswith(".csv"):
        for enc in ["utf-8", "latin1", "cp1252"]:
            uploaded_file.seek(0)
            try:
                return pd.read_csv(uploaded_file, encoding=enc)
            except Exception:
                pass
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file)
    if suffix.endswith(".xlsx") or suffix.endswith(".xls"):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file)
    raise ValueError("Formato não suportado. Envie CSV, XLSX ou XLS.")


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]

    alias_map = {
        "usuario": "nome",
        "paciente": "nome",
        "microarea": "micro_area",
        "ine": "equipe_ine",
        "equipe_saude": "equipe",
        "nome_equipe": "equipe",
        "ubs": "unidade",
        "nome_ubs": "unidade",
        "sexo_paciente": "sexo",
        "data_de_nascimento": "data_nascimento",
        "dt_nascimento": "data_nascimento",
        "nascimento": "data_nascimento",
        "logradouro": "endereco",
        "endereco": "endereco",
        "endereco_paciente": "endereco",
        "localizacao": "endereco",
    }
    for old, new in alias_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    if "idade" in df.columns:
        df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
    else:
        df["idade"] = np.nan

    for c in ["nome", "equipe", "unidade", "tipo_equipe", "sexo", "cns", "cpf", "micro_area", "equipe_ine", "data_nascimento", "endereco"]:
        if c not in df.columns:
            df[c] = ""

    if "data_nascimento" in df.columns:
        dt = pd.to_datetime(df["data_nascimento"], errors="coerce")
        formatted = dt.dt.strftime("%d/%m/%Y")
        original = df["data_nascimento"].astype(str)
        df["data_nascimento"] = np.where(dt.notna(), formatted, original)

    df["faixa_etaria"] = df["idade"].apply(faixa_etaria)
    return df


# ---------------------
# Painel de boas práticas
# ---------------------

def build_good_practices_df(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    rows = []
    weights = spec.weights or {}
    non_conditionals = spec.non_conditionals or {}
    applicability = spec.applicability or {}

    for col, peso in weights.items():
        if col not in df.columns:
            continue
        aplicavel = pd.Series(True, index=df.index)
        if col in applicability:
            aplicavel &= to_bool(df[applicability[col]])
        if col in non_conditionals:
            aplicavel &= ~non_conditionals[col](df).fillna(False).astype(bool)

        total_aplicavel = int(aplicavel.sum())
        realizados = int((aplicavel & to_bool(df[col])).sum())
        nao_realizados = max(total_aplicavel - realizados, 0)
        perc = round((realizados / total_aplicavel) * 100, 1) if total_aplicavel else 0.0

        rows.append({
            "boa_pratica": label_boa_pratica(spec.code, col),
            "coluna": col,
            "peso": peso,
            "realizados": realizados,
            "percentual_realizado": perc,
            "nao_realizados": nao_realizados,
        })

    return pd.DataFrame(rows)


def render_good_practices(df: pd.DataFrame, spec: IndicatorSpec):
    bp_df = build_good_practices_df(df, spec)
    st.markdown("### Cumprimento das boas práticas")

    st.markdown(
        """
        <style>
        .bp-card {
            background: linear-gradient(180deg, #f7fbfb 0, #eef7f7 100%);
            border: 1px solid #d9ecec;
            border-radius: 18px;
            padding: 18px 18px 10px 18px;
            box-shadow: 0 8px 26px rgba(1,105,111,0.08);
            margin-bottom: 12px;
        }
        .bp-title {
            font-size: 1.12rem;
            font-weight: 700;
            color: #0f3638;
            margin-bottom: 6px;
        }
        .bp-sub {
            font-size: 0.93rem;
            color: #476466;
            margin-bottom: 14px;
        }
        </style>
        <div class="bp-card">
            <div class="bp-title">Painel analítico das boas práticas</div>
            <div class="bp-sub">Resumo por prática com peso, quantidade realizada, percentual realizado e quantidade pendente.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_table, col_chart = st.columns([1.45, 1])

    view = bp_df.copy()
    view["percentual_realizado"] = view["percentual_realizado"].map(lambda x: f"{x:.1f}%")
    col_table.dataframe(
        view[["boa_pratica", "peso", "realizados", "percentual_realizado", "nao_realizados"]],
        use_container_width=True,
        hide_index=True,
    )

    if not bp_df.empty:
        fig = px.bar(
            bp_df.sort_values("percentual_realizado", ascending=True),
            x="percentual_realizado",
            y="boa_pratica",
            orientation="h",
            text="percentual_realizado",
            color="percentual_realizado",
            color_continuous_scale=["#cfe8e8", "#4f98a3", "#01696f"],
            title="Percentual realizado por boa prática",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            height=max(380, len(bp_df) * 72),
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        col_chart.plotly_chart(fig, use_container_width=True)


# ---------------------
# Resumo executivo
# ---------------------

def indicator_summary(df: pd.DataFrame, spec: IndicatorSpec) -> Dict[str, float]:
    if spec.type == "percentual":
        denom = int(to_bool(df[spec.denominator_col]).sum()) if len(df) else 0
        numer = int((to_bool(df[spec.denominator_col]) & to_bool(df[spec.numerator_col])).sum()) if len(df) else 0
        resultado = round((numer / denom) * 100, 1) if denom else 0.0
        return {
            "entidades": len(df),
            "denominador": denom,
            "numerador": numer,
            "resultado": resultado,
            "classificacao": classificar_score(resultado),
        }
    return {
        "entidades": len(df),
        "score_medio": round(df["score"].mean(), 1) if len(df) else 0.0,
        "score_max": round(df["score"].max(), 1) if len(df) else 0.0,
        "com_pendencias": int((df["pendencias"] > 0).sum()) if len(df) else 0,
        "classificacao": classificar_score(round(df["score"].mean(), 1) if len(df) else 0.0),
    }


def render_summary(df: pd.DataFrame, spec: IndicatorSpec):
    summary = indicator_summary(df, spec)

    st.markdown(f"## {spec.code} — {spec.name}")
    st.caption("Desempenho categorizado em Ótimo, Bom, Suficiente e Regular conforme notas metodológicas.")

    st.markdown(
        """
        <style>
        .metric-card-wrap {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 16px;
            margin: 12px 0 24px 0;
        }
        .metric-card {
            background: linear-gradient(180deg, #f8fbfb 0, #eef6f6 100%);
            border: 1px solid rgba(1,105,111,.12);
            border-radius: 18px;
            padding: 18px 18px 16px 18px;
            box-shadow: 0 8px 24px rgba(1,105,111,.08);
        }
        .metric-label {
            font-size: .82rem;
            color: #4a6668;
            margin-bottom: 6px;
            font-weight: 600;
        }
        .metric-value {
            font-size: 1.9rem;
            color: #0f3638;
            font-weight: 800;
            line-height: 1.1;
        }
        .metric-help {
            font-size: .78rem;
            color: #6f7f80;
            margin-top: 6px;
        }
        @media (max-width: 900px) {
            .metric-card-wrap { grid-template-columns: 1fr 1fr; }
        }
        @media (max-width: 640px) {
            .metric-card-wrap { grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if spec.type == "percentual":
        cards = [
            ("Total de Pacientes", summary["entidades"], spec.entity_label),
            ("Score", f"{summary['resultado']:.1f}", "resultado médio do indicador"),
            ("Desempenho", summary["classificacao"], "classificação metodológica (0-100 pontos)"),
        ]
    else:
        cards = [
            ("Total de Pacientes", summary["entidades"], spec.entity_label),
            ("Score", summary["score_medio"], "score médio (0-100 pontos)"),
            ("Desempenho", summary["classificacao"], "classificação metodológica (0-100 pontos)"),
        ]

    html_cards = "<div class='metric-card-wrap'>" + "".join(
        f"<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        f"<div class='metric-help'>{helptext}</div>"
        f"</div>"
        for (label, value, helptext) in cards
    ) + "</div>"
    st.markdown(html_cards, unsafe_allow_html=True)


# ---------------------
# Filtros
# ---------------------

def apply_filters(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    with st.sidebar:
        st.subheader("Filtros")
        equipes = sorted(x for x in df["equipe"].dropna().astype(str).unique() if x)
        unidades = sorted(x for x in df["unidade"].dropna().astype(str).unique() if x)
        microareas = sorted(x for x in df["micro_area"].dropna().astype(str).unique() if x)
        faixas = sorted(x for x in df["faixa_etaria"].dropna().astype(str).unique() if x)

        eq_sel = st.multiselect("Por equipe", equipes)
        un_sel = st.multiselect("Por unidade", unidades)
        ma_sel = st.multiselect("Por microárea", microareas)
        fx_sel = st.multiselect("Por faixa etária", faixas)

        pendencias_opts = ["Todos"]
        weight_cols = list((spec.weights or {}).keys())
        if weight_cols:
            pendencias_opts += ["Sem pendências"] + weight_cols
        pend_sel = st.selectbox("Por pendências", pendencias_opts)

    out = df.copy()
    if eq_sel:
        out = out[out["equipe"].astype(str).isin(eq_sel)]
    if un_sel:
        out = out[out["unidade"].astype(str).isin(un_sel)]
    if ma_sel:
        out = out[out["micro_area"].astype(str).isin(ma_sel)]
    if fx_sel:
        out = out[out["faixa_etaria"].astype(str).isin(fx_sel)]

    if pend_sel == "Sem pendências" and "pendencias" in out.columns:
        out = out[out["pendencias"] == 0]
    elif pend_sel in weight_cols:
        col = pend_sel
        if col in out.columns:
            out = out[~to_bool(out[col])]  # pacientes com a boa prática não realizada

    return out


# ---------------------
# Exportação
# ---------------------

def export_results(df: pd.DataFrame):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar resultado em CSV", csv, "saude360_resultado.csv", "text/csv")

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="resultado")
    xlsx = output.getvalue()
    st.download_button(
        "Baixar resultado em Excel",
        xlsx,
        "saude360_resultado.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ---------------------
# App principal (layout único)
# ---------------------

def main():
    st.title("Saúde 360 APS")
    st.caption("Leitura de uma planilha por vez, com painel único por indicador.")

    with st.sidebar:
        st.header("Configuração")
        indicator_code = st.selectbox(
            "Indicador",
            list(INDICATORS.keys()),
            format_func=lambda x: f"{x} - {INDICATORS[x].name}",
        )
        data_mode = st.radio("Origem dos dados", ["Enviar planilha", "Usar dados de demonstração"], index=0)

    spec = INDICATORS[indicator_code]
    st.info(spec.description)

    if data_mode == "Usar dados de demonstração":
        st.warning("Dados de demonstração ainda não configurados para todos os indicadores. Use uma planilha real.")
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("Envie a planilha .csv, .xls, .xlsx referente a ESTE indicador", type=["csv", "xls", "xlsx"])

    if uploaded_file is None:
        st.warning("Envie uma planilha para continuar.")
        return

    try:
        raw_df = read_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler arquivo: {e}")
        return

    df = preprocess_df(raw_df)
    if spec.type == "percentual":
        df = calculate_percent_indicator(df, spec)
    else:
        df = calculate_score_indicator(df, spec)

    df = apply_filters(df, spec)

    render_summary(df, spec)
    render_good_practices(df, spec)

    st.markdown("### Lista nominal")
    nominal_cols = [
        "nome",
        "data_nascimento",
        "idade",
        "faixa_etaria",
        "endereco",
        "sexo",
        "micro_area",
        "unidade",
        "equipe",
        "cns",
        "cpf",
        "score" if "score" in df.columns else None,
        "classificacao" if "classificacao" in df.columns else None,
    ]
    nominal_cols = [c for c in nominal_cols if c in df.columns]
    st.dataframe(df[nominal_cols], use_container_width=True, height=520)

    export_results(df)


if __name__ == "__main__":
    main()
