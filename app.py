import io
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

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


def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 1.5rem;}
        .indicator-header {
            background: linear-gradient(135deg, #0f4c5c 0%, #1d6f74 100%);
            color: white;
            border-radius: 18px;
            padding: 20px 22px;
            margin: 8px 0 18px 0;
            box-shadow: 0 10px 24px rgba(15, 76, 92, .16);
        }
        .indicator-kicker {
            font-size: .82rem;
            text-transform: uppercase;
            letter-spacing: .08em;
            opacity: .84;
            margin-bottom: 6px;
            font-weight: 700;
        }
        .indicator-title {
            font-size: 1.85rem;
            line-height: 1.12;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .indicator-desc {
            font-size: .95rem;
            line-height: 1.45;
            opacity: .95;
        }
        .metric-card-wrap {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
            margin: 6px 0 18px 0;
        }
        .metric-card {
            background: linear-gradient(180deg, #fcfefe 0%, #eef6f6 100%);
            border: 1px solid rgba(1, 105, 111, .12);
            border-radius: 18px;
            padding: 18px;
            box-shadow: 0 8px 22px rgba(1, 105, 111, .07);
        }
        .metric-label {
            font-size: .83rem;
            color: #4a6668;
            margin-bottom: 8px;
            font-weight: 700;
        }
        .metric-value {
            font-size: 1.95rem;
            line-height: 1.06;
            color: #0f3638;
            font-weight: 800;
            margin-bottom: 6px;
        }
        .metric-help {
            font-size: .78rem;
            color: #708284;
        }
        @media (max-width: 980px) {
            .metric-card-wrap {grid-template-columns: repeat(2, minmax(0,1fr));}
        }
        @media (max-width: 640px) {
            .metric-card-wrap {grid-template-columns: 1fr;}
            .indicator-title {font-size: 1.45rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_col(name: str) -> str:
    name = str(name).strip().lower()
    name = re.sub(r"[^a-z0-9à-ÿ]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name


def to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    vals = series.astype(str).str.strip().str.lower()
    return vals.isin(["1", "true", "sim", "s", "x", "ok", "yes"])


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


@dataclass
class IndicatorSpec:
    code: str
    name: str
    type: str
    description: str
    weights: Optional[Dict[str, int]] = None
    labels: Optional[Dict[str, str]] = None
    non_conditionals: Optional[Dict[str, Callable[[pd.DataFrame], pd.Series]]] = None
    applicability: Optional[Dict[str, str]] = None
    numerator_col: Optional[str] = None
    denominator_col: Optional[str] = None
    entity_label: str = "pessoas"


INDICATORS: Dict[str, IndicatorSpec] = {
    "C1": IndicatorSpec(
        code="C1",
        name="Mais acesso na APS",
        type="percentual",
        description="Percentual de atendimentos programados em relação ao total de atendimentos válidos.",
        numerator_col="demanda_programada",
        denominator_col="atendimento_valido",
        entity_label="atendimentos",
    ),
    "C2": IndicatorSpec(
        code="C2",
        name="Cuidado no desenvolvimento infantil",
        type="score",
        description="Monitoramento da criança com base em boas práticas registradas.",
        weights={"consulta_ok": 20, "vacina_ok": 20, "peso_altura_ok": 20, "visita_ok": 20, "desenvolvimento_ok": 20},
        labels={"consulta_ok": "Consulta", "vacina_ok": "Vacinação", "peso_altura_ok": "Peso e altura", "visita_ok": "Visita domiciliar", "desenvolvimento_ok": "Marco do desenvolvimento"},
        non_conditionals={"visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)},
        entity_label="crianças",
    ),
    "C3": IndicatorSpec(
        code="C3",
        name="Cuidado na gestação e puerpério",
        type="score",
        description="Pontuação por gestante/puérpera até 100 pontos.",
        weights={"pre_natal_12s_ok": 10, "consultas_gest_ok": 9, "pa_ok": 9, "antropometria_ok": 9, "visitas_gest_ok": 9, "dtpa_ok": 9, "tri1_ok": 9, "tri3_ok": 9, "puerperio_consulta_ok": 9, "puerperio_visita_ok": 9, "odonto_ok": 9},
        labels={"pre_natal_12s_ok": "1ª consulta até 12 semanas", "consultas_gest_ok": "7 consultas na gestação", "pa_ok": "Aferições de pressão arterial", "antropometria_ok": "Registros de peso e altura", "visitas_gest_ok": "3 visitas na gestação", "dtpa_ok": "Vacina dTpa", "tri1_ok": "Exames do 1º trimestre", "tri3_ok": "Exames do 3º trimestre", "puerperio_consulta_ok": "Consulta no puerpério", "puerperio_visita_ok": "Visita no puerpério", "odonto_ok": "Atividade de saúde bucal"},
        non_conditionals={"visitas_gest_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index), "puerperio_visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)},
        entity_label="gestantes/puérperas",
    ),
    "C4": IndicatorSpec(
        code="C4",
        name="Cuidado da pessoa com diabetes",
        type="score",
        description="Pontuação por pessoa com diabetes até 100 pontos.",
        weights={"consulta_ok": 20, "hba1c_ok": 20, "solicitacao_ok": 15, "pes_ok": 15, "retina_ok": 15, "visita_ok": 15},
        labels={"consulta_ok": "Consulta", "hba1c_ok": "Hemoglobina glicada", "solicitacao_ok": "Solicitação de exames", "pes_ok": "Avaliação dos pés", "retina_ok": "Avaliação de retina", "visita_ok": "Visita domiciliar"},
        non_conditionals={"visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)},
        entity_label="pessoas com diabetes",
    ),
    "C5": IndicatorSpec(
        code="C5",
        name="Cuidado da pessoa com hipertensão",
        type="score",
        description="Pontuação por pessoa com hipertensão até 100 pontos.",
        weights={"consulta_ok": 25, "pa_ok": 25, "antropometria_ok": 25, "visita_ok": 25},
        labels={"consulta_ok": "Consulta", "pa_ok": "Pressão arterial", "antropometria_ok": "Peso e altura", "visita_ok": "Visita domiciliar"},
        non_conditionals={"visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)},
        entity_label="pessoas com hipertensão",
    ),
    "C6": IndicatorSpec(
        code="C6",
        name="Cuidado da pessoa idosa",
        type="score",
        description="Pontuação por pessoa idosa até 100 pontos.",
        weights={"consulta_ok": 25, "antropometria_ok": 25, "visitas_ok": 25, "influenza_ok": 25},
        labels={"consulta_ok": "Consulta", "antropometria_ok": "Peso e altura", "visitas_ok": "Visitas domiciliares", "influenza_ok": "Vacina influenza"},
        non_conditionals={"visitas_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)},
        entity_label="pessoas idosas",
    ),
    "C7": IndicatorSpec(
        code="C7",
        name="Cuidado da mulher na prevenção do câncer",
        type="score",
        description="Pontuação por mulher elegível conforme faixa etária e práticas aplicáveis.",
        weights={"colo_utero_ok": 20, "hpv_ok": 30, "saude_reprodutiva_ok": 30, "mama_ok": 20},
        labels={"colo_utero_ok": "Citopatológico do colo do útero", "hpv_ok": "Vacinação HPV", "saude_reprodutiva_ok": "Saúde sexual e reprodutiva", "mama_ok": "Mamografia"},
        applicability={"colo_utero_ok": "colo_utero_aplicavel", "hpv_ok": "hpv_aplicavel", "saude_reprodutiva_ok": "saude_reprodutiva_aplicavel", "mama_ok": "mama_aplicavel"},
        entity_label="mulheres",
    ),
}


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
        "endereço": "endereco",
        "endereco_paciente": "endereco",
        "localizacao": "endereco",
        "rua": "endereco",
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
        df["data_nascimento"] = np.where(dt.notna(), dt.dt.strftime("%d/%m/%Y"), df["data_nascimento"].astype(str))

    df["faixa_etaria"] = df["idade"].apply(faixa_etaria)
    return df


def create_demo_dataframe(indicator_code: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 120
    base = pd.DataFrame({
        "nome": [f"Paciente {i+1}" for i in range(n)],
        "data_nascimento": pd.date_range("1955-01-01", periods=n, freq="120D").strftime("%d/%m/%Y"),
        "endereco": [f"Rua {i+10}, Bairro {((i % 8) + 1)}" for i in range(n)],
        "equipe": rng.choice(["ESF Centro", "ESF Vila Barreto", "ESF Nova Mairinque", "EAP Rural"], n),
        "unidade": rng.choice(["UBS Central", "UBS Três Lagos", "UBS Jardim Cruzeiro"], n),
        "tipo_equipe": rng.choice(["70", "76"], n, p=[0.78, 0.22]),
        "idade": rng.integers(0, 89, n),
        "sexo": rng.choice(["F", "M"], n, p=[0.62, 0.38]),
        "micro_area": rng.choice(["MA1", "MA2", "MA3", "MA4"], n),
    })

    spec = INDICATORS[indicator_code]
    if spec.type == "percentual":
        base["atendimento_valido"] = True
        base["demanda_programada"] = rng.choice([True, False], n, p=[0.56, 0.44])
        return base

    for col in spec.weights.keys():
        base[col] = rng.choice([True, False], n, p=[0.68, 0.32])

    if indicator_code == "C7":
        base["sexo"] = "F"
        base["idade"] = rng.integers(9, 70, n)
        base["colo_utero_aplicavel"] = base["idade"].between(25, 64)
        base["hpv_aplicavel"] = base["idade"].between(9, 14)
        base["saude_reprodutiva_aplicavel"] = base["idade"].between(25, 64)
        base["mama_aplicavel"] = base["idade"].between(50, 69)

    return base


def template_columns(indicator_code: str) -> List[str]:
    common = ["nome", "data_nascimento", "endereco", "cns", "cpf", "idade", "sexo", "equipe", "equipe_ine", "tipo_equipe", "unidade", "micro_area"]
    spec = INDICATORS[indicator_code]
    if spec.type == "percentual":
        return common + ["atendimento_valido", "demanda_programada"]
    cols = common + list(spec.weights.keys())
    if spec.applicability:
        cols += list(spec.applicability.values())
    return cols


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="template")
    return output.getvalue()


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
        }
    return {
        "entidades": len(df),
        "score_medio": round(df["score"].mean(), 1) if len(df) else 0.0,
        "score_max": round(df["score"].max(), 1) if len(df) else 0.0,
        "com_pendencias": int((df["pendencias"] > 0).sum()) if len(df) else 0,
    }


def render_indicator_header(spec: IndicatorSpec):
    st.markdown(
        f"""
        <div class="indicator-header">
            <div class="indicator-kicker">Indicador avaliado</div>
            <div class="indicator-title">{spec.code} — {spec.name}</div>
            <div class="indicator-desc">{spec.description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary(df: pd.DataFrame, spec: IndicatorSpec):
    summary = indicator_summary(df, spec)
    if spec.type == "percentual":
        cards = [
            ("Total de Pacientes", summary["entidades"], spec.entity_label),
            ("Score", f'{summary["resultado"]}%', "resultado do indicador"),
            ("Acompanhados", summary["numerador"], "registros no numerador"),
            ("Com pendências", max(summary["denominador"] - summary["numerador"], 0), "fora do numerador"),
        ]
    else:
        cards = [
            ("Total de Pacientes", summary["entidades"], spec.entity_label),
            ("Score", summary["score_medio"], "score médio"),
            ("Acompanhados", int((df["pendencias"] == 0).sum()) if len(df) else 0, "sem pendências"),
            ("Com pendências", summary["com_pendencias"], "com pelo menos uma pendência"),
        ]
    html = '<div class="metric-card-wrap">' + ''.join([
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-help">{help_text}</div></div>'
        for label, value, help_text in cards
    ]) + '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_percentual_panel(df: pd.DataFrame, spec: IndicatorSpec):
    team = (
        df.assign(_num=to_bool(df[spec.numerator_col]), _den=to_bool(df[spec.denominator_col]))
        .groupby("equipe", dropna=False)
        .agg(numerador=("_num", "sum"), denominador=("_den", "sum"))
        .reset_index()
    )
    team["resultado"] = np.where(team["denominador"] > 0, (team["numerador"] / team["denominador"]) * 100, 0)
    fig = px.bar(
        team.sort_values("resultado", ascending=False),
        x="equipe",
        y="resultado",
        color="resultado",
        color_continuous_scale="Tealgrn",
        title="Resultado por equipe",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(team, use_container_width=True)


def render_score_panel(df: pd.DataFrame, spec: IndicatorSpec):
    c1, c2 = st.columns(2)
    team = df.groupby("equipe", dropna=False).agg(score_medio=("score", "mean"), total=("nome", "count")).reset_index()
    unit = df.groupby("unidade", dropna=False).agg(score_medio=("score", "mean"), total=("nome", "count")).reset_index()
    fig_team = px.bar(team.sort_values("score_medio", ascending=False), x="equipe", y="score_medio", color="score_medio", color_continuous_scale="Tealgrn", title="Score médio por equipe")
    fig_unit = px.bar(unit.sort_values("score_medio", ascending=False), x="unidade", y="score_medio", color="score_medio", color_continuous_scale="Tealgrn", title="Score médio por unidade")
    c1.plotly_chart(fig_team, use_container_width=True)
    c2.plotly_chart(fig_unit, use_container_width=True)

    c3, c4 = st.columns(2)
    cls = df["classificacao"].value_counts().rename_axis("classificacao").reset_index(name="total")
    fig_cls = px.pie(
        cls,
        names="classificacao",
        values="total",
        title="Classificação geral",
        color="classificacao",
        color_discrete_map={"Ótimo": "#1f7a4d", "Bom": "#2e8b8b", "Suficiente": "#d19900", "Regular": "#a13544"},
    )
    c3.plotly_chart(fig_cls, use_container_width=True)

    pend = df[[c for c in (spec.weights or {}).keys() if c in df.columns]].copy()
    if not pend.empty:
        pend = pend.apply(to_bool)
        labels = spec.labels or {}
        cumprimento = pd.DataFrame({
            "boa_pratica": [labels.get(col, col) for col in pend.columns],
            "realizaram": pend.sum().values,
            "nao_realizaram": (~pend).sum().values,
        }).sort_values("nao_realizaram", ascending=False)
        fig_pend = px.bar(
            cumprimento,
            x="boa_pratica",
            y=["realizaram", "nao_realizaram"],
            barmode="group",
            title="Cumprimento por boa prática",
        )
        c4.plotly_chart(fig_pend, use_container_width=True)
        st.subheader("Cumprimento por boa prática")
        st.dataframe(cumprimento, use_container_width=True)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.subheader("Filtros")
        equipes = sorted([x for x in df["equipe"].dropna().astype(str).unique() if x])
        unidades = sorted([x for x in df["unidade"].dropna().astype(str).unique() if x])
        tipos = sorted([x for x in df["tipo_equipe"].dropna().astype(str).unique() if x])
        eq_sel = st.multiselect("Equipe", equipes)
        un_sel = st.multiselect("Unidade", unidades)
        tp_sel = st.multiselect("Tipo de equipe", tipos)
        idade_min, idade_max = st.slider("Faixa etária", 0, 100, (0, 100))
    out = df.copy()
    if eq_sel:
        out = out[out["equipe"].astype(str).isin(eq_sel)]
    if un_sel:
        out = out[out["unidade"].astype(str).isin(un_sel)]
    if tp_sel:
        out = out[out["tipo_equipe"].astype(str).isin(tp_sel)]
    out = out[(out["idade"].fillna(-1) >= idade_min) & (out["idade"].fillna(999) <= idade_max)]
    return out


def run_indicator(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    return calculate_percent_indicator(df, spec) if spec.type == "percentual" else calculate_score_indicator(df, spec)


def export_results(df: pd.DataFrame):
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar resultado em CSV", csv, "resultado_saude360.csv", "text/csv")
    xlsx = to_excel_bytes(df)
    st.download_button("Baixar resultado em Excel", xlsx, "resultado_saude360.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def main():
    inject_css()
    st.title("Saúde 360 APS")
    st.caption("Dashboard modular para leitura de planilhas e monitoramento dos 7 indicadores do Saúde 360.")

    with st.sidebar:
        st.header("Configuração")
        indicator_code = st.selectbox("Indicador", list(INDICATORS.keys()), format_func=lambda x: f"{x} — {INDICATORS[x].name}")
        data_mode = st.radio("Origem dos dados", ["Enviar planilha", "Usar dados de demonstração"])

    spec = INDICATORS[indicator_code]

    with st.expander("Modelo de colunas esperado"):
        cols = template_columns(indicator_code)
        st.code("\n".join(cols), language="text")
        st.download_button(
            "Baixar planilha-modelo",
            to_excel_bytes(pd.DataFrame(columns=cols)),
            f"template_{indicator_code}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if data_mode == "Usar dados de demonstração":
        raw_df = create_demo_dataframe(indicator_code)
    else:
        uploaded_file = st.file_uploader("Envie a planilha (.csv, .xls, .xlsx)", type=["csv", "xls", "xlsx"])
        if uploaded_file is None:
            st.warning("Envie uma planilha ou use os dados de demonstração.")
            st.stop()
        try:
            raw_df = read_uploaded_file(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            st.stop()

    df = preprocess_df(raw_df)
    df = run_indicator(df, spec)
    df = apply_filters(df)

    render_indicator_header(spec)
    render_summary(df, spec)

    tab1, tab2, tab3, tab4 = st.tabs(["Painel", "Nominal", "Equipes", "Qualidade dos dados"])

    with tab1:
        if spec.type == "percentual":
            render_percentual_panel(df, spec)
        else:
            render_score_panel(df, spec)

    with tab2:
        nominal_cols = [c for c in ["nome", "data_nascimento", "idade", "endereco", "sexo", "unidade", "equipe", "micro_area", "score", "pendencias", "classificacao"] if c in df.columns]
        if spec.type == "percentual":
            nominal_cols = [c for c in ["nome", "data_nascimento", "idade", "endereco", "sexo", "unidade", "equipe", spec.numerator_col, spec.denominator_col] if c in df.columns]
        st.dataframe(df[nominal_cols], use_container_width=True, height=520)
        export_results(df)

    with tab3:
        if spec.type == "percentual":
            team = df.assign(_num=to_bool(df[spec.numerator_col]), _den=to_bool(df[spec.denominator_col])).groupby(["unidade", "equipe"], dropna=False).agg(numerador=("_num", "sum"), denominador=("_den", "sum")).reset_index()
            team["resultado"] = np.where(team["denominador"] > 0, (team["numerador"] / team["denominador"]) * 100, 0)
        else:
            team = df.groupby(["unidade", "equipe"], dropna=False).agg(total=("nome", "count"), score_medio=("score", "mean"), com_pendencias=("pendencias", lambda s: int((s > 0).sum()))).reset_index()
        st.dataframe(team, use_container_width=True, height=520)

    with tab4:
        quality = pd.DataFrame({
            "coluna": df.columns,
            "nulos": [int(df[c].isna().sum()) for c in df.columns],
            "vazios": [int((df[c].astype(str).str.strip() == "").sum()) for c in df.columns],
        }).sort_values(["nulos", "vazios"], ascending=False)
        st.dataframe(quality, use_container_width=True, height=520)
        st.write("Prévia dos dados processados")
        st.dataframe(df.head(20), use_container_width=True)

    st.success("Dados processados com sucesso.")


if __name__ == "__main__":
    main()
