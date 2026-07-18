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
        "solicitacao_ok": "Registrar solicitação ou avaliação de hemoglobina glicada nos últimos 12 meses",
        "retina_ok": "Realizar avaliação de retina conforme linha de cuidado local",
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
    "C2": {
        "consulta_ok": "Registrar consulta de acompanhamento da criança conforme período de monitoramento",
        "vacina_ok": "Registrar situação vacinal adequada para a faixa etária",
        "peso_altura_ok": "Registrar peso e altura conforme acompanhamento do desenvolvimento infantil",
        "visita_ok": "Realizar visitas domiciliares conforme critério do indicador",
        "desenvolvimento_ok": "Registrar avaliação do desenvolvimento infantil no período",
    },
    "C3": {
        "pre_natal_12s_ok": "Iniciar o pré-natal até a 12ª semana de gestação",
        "consultas_gest_ok": "Realizar o número mínimo de consultas de acompanhamento da gestação",
        "pa_ok": "Registrar aferição de pressão arterial durante o acompanhamento",
        "antropometria_ok": "Registrar avaliação antropométrica durante a gestação",
        "visitas_gest_ok": "Realizar visitas domiciliares durante a gestação conforme critério do indicador",
        "dtpa_ok": "Registrar aplicação da vacina dTpa no período recomendado",
        "tri1_ok": "Realizar exames do primeiro trimestre no período oportuno",
        "tri3_ok": "Realizar exames do terceiro trimestre no período oportuno",
        "puerperio_consulta_ok": "Realizar consulta puerperal no período recomendado",
        "puerperio_visita_ok": "Realizar visita domiciliar puerperal conforme critério do indicador",
        "odonto_ok": "Registrar atendimento odontológico durante o acompanhamento",
    },
    "C7": {
        "colo_utero_ok": "Realizar exame citopatológico do colo do útero para mulheres na faixa aplicável",
        "hpv_ok": "Registrar vacinação contra HPV para meninas na faixa etária aplicável",
        "saude_reprodutiva_ok": "Registrar ação de saúde sexual e reprodutiva para mulheres na faixa aplicável",
        "mama_ok": "Registrar mamografia para mulheres na faixa etária aplicável",
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
    "C1": IndicatorSpec("C1", "Mais acesso na APS", "percentual", "Percentual de atendimentos programados em relação ao total de atendimentos válidos.", numerator_col="demanda_programada", denominator_col="atendimento_valido", entity_label="atendimentos"),
    "C2": IndicatorSpec("C2", "Cuidado no desenvolvimento infantil", "score", "Monitoramento da criança com base em boas práticas registradas.", weights={"consulta_ok": 20, "vacina_ok": 20, "peso_altura_ok": 20, "visita_ok": 20, "desenvolvimento_ok": 20}, non_conditionals={"visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)}, entity_label="crianças"),
    "C3": IndicatorSpec("C3", "Cuidado na gestação e puerpério", "score", "Pontuação por gestante/puérpera até 100 pontos.", weights={"pre_natal_12s_ok": 10, "consultas_gest_ok": 9, "pa_ok": 9, "antropometria_ok": 9, "visitas_gest_ok": 9, "dtpa_ok": 9, "tri1_ok": 9, "tri3_ok": 9, "puerperio_consulta_ok": 9, "puerperio_visita_ok": 9, "odonto_ok": 9}, non_conditionals={"visitas_gest_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index), "puerperio_visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)}, entity_label="gestantes/puérperas"),
    "C4": IndicatorSpec("C4", "Cuidado da pessoa com diabetes", "score", "Pontuação por pessoa com diabetes até 100 pontos.", weights={"consulta_ok": 20, "hba1c_ok": 20, "pes_ok": 15, "visita_ok": 20, "pa_ok": 15, "antropometria_ok": 15}, non_conditionals={"visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)}, entity_label="pessoas com diabetes"),
    "C5": IndicatorSpec("C5", "Cuidado da pessoa com hipertensão", "score", "Pontuação por pessoa com hipertensão até 100 pontos.", weights={"consulta_ok": 25, "pa_ok": 25, "antropometria_ok": 25, "visita_ok": 25}, non_conditionals={"visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)}, entity_label="pessoas com hipertensão"),
    "C6": IndicatorSpec("C6", "Cuidado da pessoa idosa", "score", "Pontuação por pessoa idosa até 100 pontos.", weights={"consulta_ok": 25, "antropometria_ok": 25, "visitas_ok": 25, "influenza_ok": 25}, non_conditionals={"visitas_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)}, entity_label="pessoas idosas"),
    "C7": IndicatorSpec("C7", "Cuidado da mulher na prevenção do câncer", "score", "Pontuação por mulher elegível conforme faixa etária e práticas aplicáveis.", weights={"colo_utero_ok": 20, "hpv_ok": 30, "saude_reprodutiva_ok": 30, "mama_ok": 20}, applicability={"colo_utero_ok": "colo_utero_aplicavel", "hpv_ok": "hpv_aplicavel", "saude_reprodutiva_ok": "saude_reprodutiva_aplicavel", "mama_ok": "mama_aplicavel"}, entity_label="mulheres"),
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
                return pd.read_csv(uploaded_file, encoding=enc, dtype=str)
            except Exception:
                pass
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, dtype=str)
    if suffix.endswith(".xlsx") or suffix.endswith(".xls"):
        uploaded_file.seek(0)
        return pd.read_excel(uploaded_file, dtype=str)
    raise ValueError("Formato não suportado. Envie CSV, XLSX ou XLS.")


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col(c) for c in df.columns]
    alias_map = {
        "nome_completo": "nome",
        "usuario": "nome",
        "paciente": "nome",
        "microarea": "micro_area",
        "equipe_area": "equipe",
        "equipe_vinculo": "equipe_vinculo",
        "ine": "equipe_ine",
        "equipe_saude": "equipe",
        "nome_equipe": "equipe",
        "ubs": "unidade",
        "nome_ubs": "unidade",
        "sexo_paciente": "sexo",
        "data_de_nascimento": "data_nascimento",
        "dt_nascimento": "data_nascimento",
        "nascimento": "data_nascimento",
        "endereco_paciente": "endereco",
        "logradouro": "endereco",
        "cadastro_atualizado": "cadastro_atualizado",
        "data_atualizacao_cadastro": "data_atualizacao_cadastro",
    }
    for old, new in alias_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]
    for txt in ["cpf", "cns", "micro_area", "equipe_ine", "tipo_equipe", "equipe_vinculo"]:
        if txt in df.columns:
            df[txt] = df[txt].astype("string")
    if "idade" in df.columns:
        df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
    else:
        df["idade"] = np.nan
    for c in ["nome", "equipe", "unidade", "tipo_equipe", "sexo", "cns", "cpf", "micro_area", "equipe_ine", "data_nascimento", "endereco", "equipe_vinculo"]:
        if c not in df.columns:
            df[c] = ""
    for c in [x for x in ["data_nascimento", "data_atualizacao_cadastro"] if x in df.columns]:
        dt = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        original = df[c].astype(str)
        df[c] = np.where(dt.notna(), dt.dt.strftime("%d/%m/%Y"), original)
    if "consulta_medica_enfermagem" in df.columns:
        df["consulta_ok"] = to_bool(df["consulta_medica_enfermagem"])
    if "hemoglobina_glicada" in df.columns:
        df["hba1c_ok"] = to_bool(df["hemoglobina_glicada"])
        df["solicitacao_ok"] = to_bool(df["hemoglobina_glicada"])
    if "avaliacao_dos_pes" in df.columns:
        df["pes_ok"] = to_bool(df["avaliacao_dos_pes"])
    if "afericao_de_pa" in df.columns:
        df["pa_ok"] = to_bool(df["afericao_de_pa"])
    if "qtd_registros_de_peso_altura" in df.columns:
        qtd_pa = parse_count(df["qtd_registros_de_peso_altura"])
        df["peso_altura_ok"] = qtd_pa.fillna(0).ge(1)
        df["antropometria_ok"] = qtd_pa.fillna(0).ge(1)
    if "qtd_visitas_domiciliares" in df.columns:
        qtd_vis = parse_count(df["qtd_visitas_domiciliares"])
        df["visita_ok"] = qtd_vis.fillna(0).ge(1)
        df["visitas_ok"] = qtd_vis.fillna(0).ge(1)
    if "acompanhado" in df.columns:
        df["acompanhado_ok"] = to_bool(df["acompanhado"])
    if "vacina_influenza" in df.columns:
        df["influenza_ok"] = to_bool(df["vacina_influenza"])
    if "citopatologico" in df.columns:
        df["colo_utero_ok"] = to_bool(df["citopatologico"])
    if "hpv" in df.columns:
        df["hpv_ok"] = to_bool(df["hpv"])
    if "saude_sexual_reprodutiva" in df.columns:
        df["saude_reprodutiva_ok"] = to_bool(df["saude_sexual_reprodutiva"])
    if "mamografia" in df.columns:
        df["mama_ok"] = to_bool(df["mamografia"])
    idade = pd.to_numeric(df["idade"], errors="coerce")
    df["colo_utero_aplicavel"] = idade.between(25, 64, inclusive="both")
    df["hpv_aplicavel"] = idade.between(9, 14, inclusive="both")
    df["saude_reprodutiva_aplicavel"] = idade.between(25, 64, inclusive="both")
    df["mama_aplicavel"] = idade.between(50, 69, inclusive="both")
    df["faixa_etaria"] = df["idade"].apply(faixa_etaria)
    return df


def detect_indicator_from_name(filename: str) -> Optional[str]:
    n = normalize_col(filename)
    if "diabetes" in n:
        return "C4"
    if "hipertensao" in n:
        return "C5"
    if "idosa" in n or "idoso" in n:
        return "C6"
    if "cancer" in n or "mulher" in n:
        return "C7"
    if "gestante" in n or "puerpera" in n or "gestacao" in n:
        return "C3"
    if "desenvolvimento_infantil" in n or "infantil" in n:
        return "C2"
    if "acesso" in n:
        return "C1"
    return None


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
        df.to_excel(writer, index=False, sheet_name="dados")
    return output.getvalue()


def indicator_summary(df: pd.DataFrame, spec: IndicatorSpec) -> Dict[str, float]:
    if spec.type == "percentual":
        denom = int(to_bool(df[spec.denominator_col]).sum()) if len(df) else 0
        numer = int((to_bool(df[spec.denominator_col]) & to_bool(df[spec.numerator_col])).sum()) if len(df) else 0
        resultado = round((numer / denom) * 100, 1) if denom else 0.0
        return {"entidades": len(df), "denominador": denom, "numerador": numer, "resultado": resultado}
    return {"entidades": len(df), "score_medio": round(df["score"].mean(), 1) if len(df) else 0.0, "score_max": round(df["score"].max(), 1) if len(df) else 0.0, "com_pendencias": int((df["pendencias"] > 0).sum()) if len(df) else 0}


def microarea_summary(df: pd.DataFrame, spec: IndicatorSpec) -> str:
    if "micro_area" not in df.columns:
        return "Sem microárea"
    base = df[df["micro_area"].astype(str).str.strip() != ""].copy()
    if base.empty:
        return "Sem microárea"
    if spec.type == "percentual":
        agg = base.assign(_num=to_bool(base[spec.numerator_col]), _den=to_bool(base[spec.denominator_col])).groupby("micro_area", dropna=False).agg(numerador=("_num", "sum"), denominador=("_den", "sum")).reset_index()
        agg["valor"] = np.where(agg["denominador"] > 0, (agg["numerador"] / agg["denominador"]) * 100, 0)
    else:
        agg = base.groupby("micro_area", dropna=False).agg(valor=("score", "mean")).reset_index()
    if agg.empty:
        return "Sem microárea"
    best = agg.sort_values("valor", ascending=False).iloc[0]
    return f"{best['micro_area']}: {round(float(best['valor']), 1)}"


def render_summary(df: pd.DataFrame, spec: IndicatorSpec):
    summary = indicator_summary(df, spec)
    st.subheader(f"{spec.code} — {spec.name}")
    c1, c2, c3, c4 = st.columns(4)
    if spec.type == "percentual":
        c1.metric("Total", summary["entidades"])
        c2.metric("Resultado", f"{summary['resultado']}%")
        c3.metric("Melhor microárea", microarea_summary(df, spec))
        c4.metric("Pendências", max(summary["denominador"] - summary["numerador"], 0))
    else:
        c1.metric("Total", summary["entidades"])
        c2.metric("Score médio", summary["score_medio"])
        c3.metric("Melhor microárea", microarea_summary(df, spec))
        c4.metric("Com pendências", summary["com_pendencias"])


def render_charts(df: pd.DataFrame, spec: IndicatorSpec):
    if spec.type == "percentual":
        if "equipe" in df.columns:
            g = df.assign(_num=to_bool(df[spec.numerator_col]), _den=to_bool(df[spec.denominator_col])).groupby("equipe", dropna=False).agg(numerador=("_num", "sum"), denominador=("_den", "sum")).reset_index()
            g["resultado"] = np.where(g["denominador"] > 0, (g["numerador"] / g["denominador"]) * 100, 0)
            fig = px.bar(g.sort_values("resultado", ascending=False), x="equipe", y="resultado", title="Resultado por equipe")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        return
    col1, col2 = st.columns(2)
    if "equipe" in df.columns:
        g1 = df.groupby("equipe", dropna=False).agg(score_medio=("score", "mean")).reset_index().sort_values("score_medio", ascending=False)
        fig1 = px.bar(g1, x="equipe", y="score_medio", title="Score médio por equipe", color="score_medio", color_continuous_scale="Teal")
        fig1.update_layout(height=420, coloraxis_showscale=False)
        col1.plotly_chart(fig1, use_container_width=True)
    if "classificacao" in df.columns:
        g2 = df.groupby("classificacao", dropna=False).size().reset_index(name="qtd")
        fig2 = px.pie(g2, names="classificacao", values="qtd", title="Classificação")
        fig2.update_layout(height=420)
        col2.plotly_chart(fig2, use_container_width=True)


def build_good_practices_df(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    rows = []
    weights = spec.weights or {}
    applicability = spec.applicability or {}
    non_conditionals = spec.non_conditionals or {}
    for col, peso in weights.items():
        if col not in df.columns:
            continue
        aplicavel = pd.Series(True, index=df.index)
        if col in applicability and applicability[col] in df.columns:
            aplicavel &= to_bool(df[applicability[col]])
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
    if spec.type != "score" or not spec.weights:
        return
    bp_df = build_good_practices_df(df, spec)
    if bp_df.empty:
        return
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
    col_tabela, col_grafico = st.columns([1.45, 1])
    styled = bp_df.style.format({"% Realizado": "{:.1f}%"}).background_gradient(subset=["% Realizado"], cmap="BuGn").background_gradient(subset=["Não realizado"], cmap="OrRd")
    col_tabela.dataframe(styled, use_container_width=True, hide_index=True)
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


def render_table(df: pd.DataFrame, spec: IndicatorSpec):
    base_cols = [c for c in ["nome", "idade", "faixa_etaria", "equipe", "micro_area", "unidade", "cpf", "cns"] if c in df.columns]
    metric_cols = [spec.numerator_col, spec.denominator_col] if spec.type == "percentual" else list((spec.weights or {}).keys()) + ["score", "pendencias", "classificacao"]
    cols = [c for c in base_cols + metric_cols if c in df.columns]
    st.markdown("### Lista nominal")
    st.dataframe(df[cols], use_container_width=True, hide_index=True)


def process_indicator(df: pd.DataFrame, indicator_code: str) -> pd.DataFrame:
    spec = INDICATORS[indicator_code]
    df = preprocess_df(df)
    if spec.type == "percentual":
        return calculate_percent_indicator(df, spec)
    return calculate_score_indicator(df, spec)


def main():
    st.title("Saúde 360 APS")
    st.caption("Dashboard multiperfil para os 7 indicadores com leitura de arquivos reais.")
    with st.sidebar:
        indicator_code = st.selectbox("Indicador", list(INDICATORS.keys()), format_func=lambda x: f"{x} — {INDICATORS[x].name}")
        uploaded = st.file_uploader("Envie CSV, XLS ou XLSX", type=["csv", "xls", "xlsx"])
        st.markdown("---")
        tpl = pd.DataFrame(columns=template_columns(indicator_code))
        st.download_button("Baixar template Excel", data=to_excel_bytes(tpl), file_name=f"template_{indicator_code}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if not uploaded:
        st.warning("Envie um arquivo para continuar.")
        return
    raw = read_uploaded_file(uploaded)
    processed = process_indicator(raw, indicator_code)
    spec = INDICATORS[indicator_code]
    with st.sidebar:
        st.markdown("### Filtros")
        if "equipe" in processed.columns:
            equipes = sorted([e for e in processed["equipe"].astype(str).dropna().unique() if e.strip()])
            equipe_sel = st.multiselect("Equipes", equipes)
            if equipe_sel:
                processed = processed[processed["equipe"].isin(equipe_sel)]
        if "micro_area" in processed.columns:
            micros = sorted([e for e in processed["micro_area"].astype(str).dropna().unique() if e.strip()])
            micro_sel = st.selectbox("Microárea", ["Todas"] + micros)
            if micro_sel != "Todas":
                processed = processed[processed["micro_area"].astype(str) == micro_sel]
        if "faixa_etaria" in processed.columns:
            faixas = sorted([e for e in processed["faixa_etaria"].astype(str).dropna().unique() if e.strip()])
            faixa_sel = st.selectbox("Faixa etária", ["Todas"] + faixas)
            if faixa_sel != "Todas":
                processed = processed[processed["faixa_etaria"].astype(str) == faixa_sel]
        if spec.type == "score" and "pendencias" in processed.columns:
            pendencia_sel = st.checkbox("Por pendências (somente quem tem pendência)")
            if pendencia_sel:
                processed = processed[processed["pendencias"] > 0]
    render_summary(processed, spec)
    render_charts(processed, spec)
    render_good_practices(processed, spec)
    st.markdown("---")
    render_table(processed, spec)
    csv_bytes = processed.to_csv(index=False).encode("utf-8-sig")
    st.download_button("Baixar CSV tratado", data=csv_bytes, file_name=f"{indicator_code.lower()}_tratado.csv", mime="text/csv")
    st.download_button("Baixar Excel tratado", data=to_excel_bytes(processed), file_name=f"{indicator_code.lower()}_tratado.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
