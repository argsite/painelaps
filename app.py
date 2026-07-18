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
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))


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


def count_to_bool(series: pd.Series) -> pd.Series:
    vals = series.astype(str).str.strip().str.lower()
    vals = vals.replace({"nan": "", "none": "", "": "0", "n": "0", "nao": "0", "não": "0"})
    vals = vals.str.replace("+", "", regex=False)
    nums = pd.to_numeric(vals, errors="coerce").fillna(0)
    return nums > 0


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
    non_conditionals: Optional[Dict[str, Callable[[pd.DataFrame], pd.Series]]] = None
    applicability: Optional[Dict[str, str]] = None
    numerator_col: Optional[str] = None
    denominator_col: Optional[str] = None
    entity_label: str = "pessoas"


INDICATORS: Dict[str, IndicatorSpec] = {
    "C1": IndicatorSpec("C1", "Mais acesso na APS", "percentual", "Percentual de atendimentos programados em relação ao total de atendimentos válidos.", numerator_col="demanda_programada", denominator_col="atendimento_valido", entity_label="atendimentos"),
    "C2": IndicatorSpec("C2", "Cuidado no desenvolvimento infantil", "score", "Monitoramento da criança com base em boas práticas registradas.", weights={"consulta_ok": 20, "vacina_ok": 20, "peso_altura_ok": 20, "visita_ok": 20, "desenvolvimento_ok": 20}, non_conditionals={"visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)}, entity_label="crianças"),
    "C3": IndicatorSpec("C3", "Cuidado na gestação e puerpério", "score", "Pontuação por gestante/puérpera até 100 pontos.", weights={"pre_natal_12s_ok": 10, "consultas_gest_ok": 9, "pa_ok": 9, "antropometria_ok": 9, "visitas_gest_ok": 9, "dtpa_ok": 9, "tri1_ok": 9, "tri3_ok": 9, "puerperio_consulta_ok": 9, "puerperio_visita_ok": 9, "odonto_ok": 9}, non_conditionals={"visitas_gest_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index), "puerperio_visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)}, entity_label="gestantes/puérperas"),
    "C4": IndicatorSpec("C4", "Cuidado da pessoa com diabetes", "score", "Pontuação por pessoa com diabetes até 100 pontos.", weights={"consulta_ok": 20, "hba1c_ok": 20, "solicitacao_ok": 15, "pes_ok": 15, "retina_ok": 15, "visita_ok": 15}, non_conditionals={"visita_ok": lambda d: d["tipo_equipe"].astype(str).eq("76") if "tipo_equipe" in d.columns else pd.Series(False, index=d.index)}, entity_label="pessoas com diabetes"),
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
        "nome_completo": "nome",
        "microarea": "micro_area",
        "equipe_area": "equipe",
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
        "endereco_paciente": "endereco",
        "localizacao": "endereco",
    }
    for old, new in alias_map.items():
        if old in df.columns and new not in df.columns:
            df[new] = df[old]

    for c in ["cpf", "cns", "micro_area", "equipe_ine", "equipe_vinculo"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    if "idade" in df.columns:
        df["idade"] = pd.to_numeric(df["idade"], errors="coerce")
    else:
        df["idade"] = np.nan

    for c in ["nome", "equipe", "unidade", "tipo_equipe", "sexo", "cns", "cpf", "micro_area", "equipe_ine", "data_nascimento", "endereco"]:
        if c not in df.columns:
            df[c] = ""

    if "data_nascimento" in df.columns:
        dt = pd.to_datetime(df["data_nascimento"], errors="coerce", dayfirst=True)
        formatted = dt.dt.strftime("%d/%m/%Y")
        original = df["data_nascimento"].astype(str)
        df["data_nascimento"] = np.where(dt.notna(), formatted, original)

    df["faixa_etaria"] = df["idade"].apply(faixa_etaria)
    return df


def apply_report_mapping(df: pd.DataFrame, indicator_code: str) -> pd.DataFrame:
    df = df.copy()

    if indicator_code == "C4":
        if "consulta_medica_enfermagem" in df.columns:
            df["consulta_ok"] = to_bool(df["consulta_medica_enfermagem"])
        if "hemoglobina_glicada" in df.columns:
            df["hba1c_ok"] = to_bool(df["hemoglobina_glicada"])
        if "avaliacao_dos_pes" in df.columns:
            df["pes_ok"] = to_bool(df["avaliacao_dos_pes"])
        if "qtd_visitas_domiciliares" in df.columns:
            df["visita_ok"] = count_to_bool(df["qtd_visitas_domiciliares"])
        if "afericao_de_pa" in df.columns and "pa_ok" not in df.columns:
            df["pa_ok"] = to_bool(df["afericao_de_pa"])
        if "qtd_registros_de_peso_altura" in df.columns and "antropometria_ok" not in df.columns:
            df["antropometria_ok"] = count_to_bool(df["qtd_registros_de_peso_altura"])
        if "solicitacao_ok" not in df.columns:
            df["solicitacao_ok"] = False
        if "retina_ok" not in df.columns:
            df["retina_ok"] = False

    if indicator_code == "C5":
        if "consulta_medica_enfermagem" in df.columns:
            df["consulta_ok"] = to_bool(df["consulta_medica_enfermagem"])
        if "afericao_de_pa" in df.columns:
            df["pa_ok"] = to_bool(df["afericao_de_pa"])
        if "qtd_registros_de_peso_altura" in df.columns:
            df["antropometria_ok"] = count_to_bool(df["qtd_registros_de_peso_altura"])
        if "qtd_visitas_domiciliares" in df.columns:
            df["visita_ok"] = count_to_bool(df["qtd_visitas_domiciliares"])

    if indicator_code == "C6":
        if "consulta_medica_enfermagem" in df.columns:
            df["consulta_ok"] = to_bool(df["consulta_medica_enfermagem"])
        if "qtd_registros_de_peso_altura" in df.columns:
            df["antropometria_ok"] = count_to_bool(df["qtd_registros_de_peso_altura"])
        if "qtd_visitas_domiciliares" in df.columns:
            df["visitas_ok"] = count_to_bool(df["qtd_visitas_domiciliares"])
        if "vacinacao_influenza" in df.columns:
            df["influenza_ok"] = to_bool(df["vacinacao_influenza"])
        elif "influenza" not in df.columns:
            df["influenza_ok"] = False

    return df


def detect_indicator_by_columns(df: pd.DataFrame) -> Optional[str]:
    cols = set(df.columns)
    if {"hemoglobina_glicada", "avaliacao_dos_pes"}.issubset(cols):
        return "C4"
    if {"afericao_de_pa", "qtd_registros_de_peso_altura"}.issubset(cols):
        return "C5"
    return None


def create_demo_dataframe(indicator_code: str) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 120
    base = pd.DataFrame({
        "nome": [f"Paciente {i+1}" for i in range(n)],
        "data_nascimento": pd.date_range("1955-01-01", periods=n, freq="120D").strftime("%d/%m/%Y"),
        "endereco": [f"Rua {i+10}, Bairro {((i%8)+1)}" for i in range(n)],
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
        return {"entidades": len(df), "denominador": denom, "numerador": numer, "resultado": resultado}
    return {
        "entidades": len(df),
        "score_medio": round(df["score"].mean(), 1) if len(df) else 0.0,
        "score_max": round(df["score"].max(), 1) if len(df) else 0.0,
        "com_pendencias": int((df["pendencias"] > 0).sum()) if len(df) else 0,
    }


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
        c3.metric("Numerador", summary["numerador"])
        c4.metric("Denominador", summary["denominador"])
    else:
        c1.metric("Total", summary["entidades"])
        c2.metric("Score médio", summary["score_medio"])
        c3.metric("Melhor microárea", microarea_summary(df, spec))
        c4.metric("Com pendências", summary["com_pendencias"])


def main():
    st.title("Saúde 360 APS")
    st.caption("Versão revisada com parser ajustado para relatórios reais de diabetes e hipertensão.")

    with st.sidebar:
        indicador = st.selectbox("Indicador", list(INDICATORS.keys()), format_func=lambda x: f"{x} — {INDICATORS[x].name}")
        uploaded = st.file_uploader("Envie CSV/XLS/XLSX", type=["csv", "xls", "xlsx"])
        usar_demo = st.checkbox("Usar dados de demonstração", value=uploaded is None)

    spec = INDICATORS[indicador]

    if usar_demo and uploaded is None:
        df = create_demo_dataframe(indicador)
        st.info("Exibindo dados de demonstração.")
    elif uploaded is not None:
        raw = read_uploaded_file(uploaded)
        df = preprocess_df(raw)
        detectado = detect_indicator_by_columns(df)
        if detectado and detectado != indicador:
            st.warning(f"O arquivo parece ser do indicador {detectado}, mas o seletor está em {indicador}.")
        df = apply_report_mapping(df, indicador)
        st.write("Colunas identificadas:", list(df.columns))
    else:
        st.stop()

    if spec.type == "percentual":
        result = calculate_percent_indicator(df, spec)
    else:
        result = calculate_score_indicator(df, spec)

    render_summary(result, spec)

    if spec.type == "score" and "score" in result.columns:
        tab1, tab2, tab3 = st.tabs(["Painel", "Dados", "Qualidade"])
        with tab1:
            if "equipe" in result.columns:
                equipe = result.groupby("equipe", dropna=False).agg(score_medio=("score", "mean"), pessoas=("nome", "count")).reset_index().sort_values("score_medio", ascending=False)
                fig = px.bar(equipe, x="equipe", y="score_medio", text="pessoas", title="Score médio por equipe")
                st.plotly_chart(fig, use_container_width=True)
            if "classificacao" in result.columns:
                fig2 = px.histogram(result, x="classificacao", color="classificacao", title="Distribuição por classificação")
                st.plotly_chart(fig2, use_container_width=True)
        with tab2:
            st.dataframe(result, use_container_width=True)
            st.download_button("Baixar CSV tratado", data=result.to_csv(index=False).encode("utf-8-sig"), file_name=f"{indicador.lower()}_tratado.csv", mime="text/csv")
        with tab3:
            qual = pd.DataFrame({
                "coluna": result.columns,
                "nulos": [int(result[c].isna().sum()) for c in result.columns],
                "preenchidos": [int(result[c].notna().sum()) for c in result.columns],
            })
            st.dataframe(qual, use_container_width=True)
    else:
        st.dataframe(result, use_container_width=True)


if __name__ == "__main__":
    main()
