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
    out = np.where(
        (pd.Series(out, index=series.index) == "")
        & (vals.str.contains(" 70") | vals.str.contains("TIPO 70")),
        "70",
        out,
    )
    return pd.Series(out, index=series.index)


# =========================
# Especificações
# =========================


@dataclass
class IndicatorSpec:
    code: str
    name: str
    type: str
    description: str
    weights: Dict[str, int] = field(default_factory=dict)
    non_conditionals: Dict[str, Callable[[pd.DataFrame], pd.Series]] = field(
        default_factory=dict
    )
    numerator_col: Optional[str] = None
    denominator_col: Optional[str] = None
    entity_label: str = "pessoas"
    applicable_age_rule: Optional[Callable[[pd.DataFrame], pd.Series]] = None


BOA_PRATICA_LABELS = {
    "C2": {
        "c2_a_ok": "A - Ter a 1ª consulta presencial realizada por médica(o) ou enfermeira(o), até o 30º dia de vida",
        "c2_b_ok": "B - Ter pelo menos 09 (nove) consultas presenciais ou remotas realizadas por médica(o) ou enfermeira(o) até dois anos de vida",
        "c2_c_ok": "C - Ter pelo menos 09 (nove) registros simultâneos de peso e altura até os dois anos de vida",
        "c2_d_ok": "D - Ter pelo menos 02 (duas) visitas domiciliares realizadas por ACS/TACS, sendo a primeira até os primeiros 30 (trinta) dias de vida e a segunda até os 06 (seis) meses de vida",
        "c2_e_ok": "E - Ter vacinas registradas com todas as doses recomendadas até os 2 anos",
    },
    "C3": {
        "c3_a_ok": "A - Ter a 1ª consulta presencial ou remota realizada por médica(o) ou enfermeira(o), até a 12ª semana de gestação.",
        "c3_b_ok": "B - Ter pelo menos 07 (sete) consultas presenciais ou remotas realizadas por médica(o) ou enfermeira(o) durante o período da gestação.",
        "c3_c_ok": "C - Ter pelo menos 07 (sete) registro de aferição de pressão arterial realizados durante o período da gestação.",
        "c3_d_ok": "D - Ter pelo menos 07 (sete) registros simultâneos de peso e altura durante o período da gestação.",
        "c3_e_ok": "E - Ter pelo menos 03 (três) visitas domiciliares realizadas por ACS/TACS, após a primeira consulta do pré-natal.",
        "c3_f_ok": "F - Ter vacina acelular contra difteria, tétano, coqueluche (dTpa) registrada a partir da 20ª semana de cada gestação.",
        "c3_g_ok": "G - Ter registro dos testes rápidos ou dos exames avaliados para sífilis, HIV e hepatites B e C realizados no 1º trimestre de cada gestação.",
        "c3_h_ok": "H - Ter registro dos testes rápidos ou dos exames avaliados para sífilis e HIV realizados no 3º trimestre de cada gestação.",
        "c3_i_ok": "I - Ter pelo menos 01 registro de consulta presencial ou remota realizada por médica(o) ou enfermeira(o) durante o puerpério.",
        "c3_j_ok": "J - Ter pelo menos 01 visita domiciliar realizada por ACS/TACS durante o puerpério.",
        "c3_k_ok": "K - Ter pelo menos 01 atividade em saúde bucal realizada por cirurgiã(ão) dentista ou técnica(o) de saúde bucal durante o período da gestação.",
    },
    "C4": {
        "c4_a_ok": "A - Ter pelo menos 01 (uma) consulta presencial ou remota realizadas por médica(o) ou enfermeira(o), nos últimos 06 (seis) meses",
        "c4_b_ok": "B - Ter pelo menos 01 (um) registro de aferição de pressão arterial realizado nos últimos 06 (seis) meses",
        "c4_c_ok": "C - Ter pelo menos 01 (um) registro simultâneos de peso e altura realizado nos últimos 12 (doze) meses",
        "c4_d_ok": "D - Ter pelo menos 02 (duas) visitas domiciliares realizadas por ACS/TACS, com intervalo mínimo de 30 (trinta) dias, nos últimos 12 (doze) meses",
        "c4_e_ok": "E - Ter pelo menos 01 (um) registro de solicitação de hemoglobina glicada realizada ou avaliada, nos últimos 12 (doze) meses",
        "c4_f_ok": "F - Ter pelo menos 01 (uma) avaliação dos pés realizada nos últimos 12 (doze) meses",
    },
    "C5": {
        "c5_a_ok": "A - Ter pelo menos 01 (uma) consulta presencial ou remota realizadas por médica(o) ou enfermeira(o), nos últimos 06 (seis) meses",
        "c5_b_ok": "B - Ter pelo menos 01 (um) registro de aferição de pressão arterial realizado nos últimos 06 (seis) meses",
        "c5_c_ok": "C - Ter pelo menos 01 (um) registro simultâneos de peso e altura realizado nos últimos 12 (doze) meses",
        "c5_d_ok": "D - Ter pelo menos 02 (duas) visitas domiciliares realizadas por ACS/TACS, com intervalo mínimo de 30 (trinta) dias, nos últimos 12 (doze) meses",
    },
    "C6": {
        "consulta_ok": "A - Ter registro de pelo menos 01 consulta presencial ou remota por profissional médica(o) ou enfermeira(o) realizada nos últimos 12 meses",
        "antropometria_ok": "B - Ter realizado pelo menos 01 (um) registro simultâneo (no mesmo dia) de peso e altura para avaliação antropométrica nos últimos 12 meses",
        "visitas_ok": "C - Ter registro de pelo menos 02 visitas domiciliares por ACS/TACS, com intervalo mínimo de 30 dias, realizadas nos últimos 12 meses",
        "influenza_ok": "D - Ter registro de 1 dose da vacina contra influenza realizada nos últimos 12 meses",
    },
    "C7": {
        "c7_a_ok": "A - Exame citopatológico (25-64 anos) ou molecular de HPV (até 60 meses)",
        "c7_b_ok": "B - Pelo menos 1 dose da vacina HPV (9-14 anos)",
        "c7_c_ok": "C - Atendimento em saúde sexual e reprodutiva nos últimos 12 meses",
        "c7_d_ok": "D - Mamografia de rastreamento (50-69 anos) realizada ou avaliada em 24 meses",
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
        description="Monitoramento da puericultura de crianças até 2 anos com base nas práticas A–E.",
        weights={
            "c2_a_ok": 20,
            "c2_b_ok": 20,
            "c2_c_ok": 20,
            "c2_d_ok": 20,
            "c2_e_ok": 20,
        },
        entity_label="crianças acompanhadas",
    ),
    "C3": IndicatorSpec(
        code="C3",
        name="Cuidado na gestação e puerpério",
        type="score",
        description="Painel operacional local para gestantes e puérperas com base nas práticas A–K.",
        weights={
            "c3_a_ok": 10,
            "c3_b_ok": 10,
            "c3_c_ok": 10,
            "c3_d_ok": 10,
            "c3_e_ok": 10,
            "c3_f_ok": 10,
            "c3_g_ok": 10,
            "c3_h_ok": 10,
            "c3_i_ok": 10,
            "c3_j_ok": 10,
            "c3_k_ok": 10,
        },
        entity_label="gestantes/puérperas",
    ),
    "C4": IndicatorSpec(
        code="C4",
        name="Cuidado da pessoa com diabetes",
        type="score",
        description="Pontuação por pessoa com diabetes até 100 pontos a partir das práticas A–F.",
        weights={
            "c4_a_ok": 16,
            "c4_b_ok": 16,
            "c4_c_ok": 16,
            "c4_d_ok": 16,
            "c4_e_ok": 18,
            "c4_f_ok": 18,
        },
        entity_label="pessoas com diabetes",
    ),
    "C5": IndicatorSpec(
        code="C5",
        name="Cuidado da pessoa com hipertensão",
        type="score",
        description="Pontuação por pessoa com hipertensão até 100 pontos a partir das práticas A–D.",
        weights={
            "c5_a_ok": 25,
            "c5_b_ok": 25,
            "c5_c_ok": 25,
            "c5_d_ok": 25,
        },
        entity_label="pessoas com hipertensão",
    ),
    "C6": IndicatorSpec(
        code="C6",
        name="Cuidado da pessoa idosa",
        type="score",
        description="Pontuação por pessoa idosa até 100 pontos.",
        weights={
            "consulta_ok": 25,
            "antropometria_ok": 25,
            "visitas_ok": 25,
            "influenza_ok": 25,
        },
        entity_label="pessoas idosas",
    ),
    "C7": IndicatorSpec(
        code="C7",
        name="Cuidado da mulher na prevenção do câncer",
        type="score",
        description="Painel operacional local para prevenção do câncer da mulher com base nas práticas A–D.",
        weights={
            "c7_a_ok": 25,
            "c7_b_ok": 25,
            "c7_c_ok": 25,
            "c7_d_ok": 25,
        },
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
    if "hipertensao" in name or {
        "afericao_de_pa",
        "qtd_registros_de_peso_altura",
    }.issubset(cols):
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
# Pré-processamento
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

    if "consulta_medica_enfermagem" in df.columns:
        df["consulta_ok"] = to_bool(df["consulta_medica_enfermagem"])
    elif "consulta" in df.columns:
        df["consulta_ok"] = to_bool(df["consulta"])
    else:
        df["consulta_ok"] = False

    if "afericao_de_pressao_arterial" in df.columns:
        df["c5_b_ok"] = to_bool(df["afericao_de_pressao_arterial"])
    elif "afericao_de_pa" in df.columns:
        df["c5_b_ok"] = to_bool(df["afericao_de_pa"])
    else:
        df["c5_b_ok"] = df.get("pa_ok", False)

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

    df["cadastro_ok"] = to_bool(df["cadastro_atualizado"]) if "cadastro_atualizado" in df.columns else False
    df["atendimento_ok"] = to_bool(df["acompanhado"]) if "acompanhado" in df.columns else df["consulta_ok"]
    df["numerador_c1"] = (df["cadastro_ok"] | df["atendimento_ok"]).astype(int)
    df["denominador_c1"] = 1

    if indicator_code == "C2" or (indicator_code is None and df["idade"].notna().any()):
        df["vacina_ok"] = to_bool(df["vacina_influenza"]) if "vacina_influenza" in df.columns else False

    df["exame_ok"] = False
    possible_exam_cols = [
        c for c in df.columns if any(k in c for k in ["exame", "teste", "hemoglobina", "citopatologico", "mamografia"])
    ]
    if possible_exam_cols:
        temp = pd.Series(False, index=df.index)
        for c in possible_exam_cols:
            temp = temp | to_bool(df[c])
        df["exame_ok"] = temp

    df["citopatologico_ok"] = False
    df["mamografia_ok"] = False
    if "citopatologico" in df.columns:
        df["citopatologico_ok"] = to_bool(df["citopatologico"])
    elif "acompanhado" in df.columns and indicator_code == "C7":
        df["citopatologico_ok"] = to_bool(df["acompanhado"])

    if "mamografia" in df.columns:
        df["mamografia_ok"] = to_bool(df["mamografia"])

    # C2
    if indicator_code == "C2":
        if "consulta_ate_30_dias" in df.columns:
            df["c2_a_ok"] = to_bool(df["consulta_ate_30_dias"])
        if "qtd_consultas" in df.columns:
            df["c2_b_ok"] = parse_count(df["qtd_consultas"]).fillna(0).ge(9)
        if "qtd_registros_de_peso_altura" in df.columns:
            df["c2_c_ok"] = parse_count(df["qtd_registros_de_peso_altura"]).fillna(0).ge(9)
        if "visita_oportuna" in df.columns:
            df["c2_d_ok"] = to_bool(df["visita_oportuna"])
        if "vacina_em_dia" in df.columns:
            df["c2_e_ok"] = to_bool(df["vacina_em_dia"])

    # C3
    if indicator_code == "C3":
        if "consulta_inicial_ate_12s" in df.columns:
            df["c3_a_ok"] = to_bool(df["consulta_inicial_ate_12s"])
        if "qtd_consultas_prenatal" in df.columns:
            df["c3_b_ok"] = parse_count(df["qtd_consultas_prenatal"]).fillna(0).ge(7)
        if "afericoes_pa" in df.columns:
            df["c3_c_ok"] = parse_count(df["afericoes_pa"]).fillna(0).ge(7)
        if "qtd_registros_de_peso_altura" in df.columns:
            df["c3_d_ok"] = parse_count(df["qtd_registros_de_peso_altura"]).fillna(0).ge(7)
        if "qtd_visitas_domiciliares" in df.columns:
            df["c3_e_ok"] = parse_count(df["qtd_visitas_domiciliares"]).fillna(0).ge(3)
        if "dtpa_ok" in df.columns:
            df["c3_f_ok"] = to_bool(df["dtpa_ok"])
        if "exames_1t_ok" in df.columns:
            df["c3_g_ok"] = to_bool(df["exames_1t_ok"])
        if "exames_3t_ok" in df.columns:
            df["c3_h_ok"] = to_bool(df["exames_3t_ok"])
        if "consulta_puerperio_ok" in df.columns:
            df["c3_i_ok"] = to_bool(df["consulta_puerperio_ok"])
        if "visita_puerperio_ok" in df.columns:
            df["c3_j_ok"] = to_bool(df["visita_puerperio_ok"])
        if "saude_bucal_ok" in df.columns:
            df["c3_k_ok"] = to_bool(df["saude_bucal_ok"])

    # C4
    if indicator_code == "C4":
        df["c4_a_ok"] = to_bool(df["consulta_medica_enfermagem"]) if "consulta_medica_enfermagem" in df.columns else df.get("consulta_ok", False)
        df["c4_b_ok"] = to_bool(df["afericao_de_pa"]) if "afericao_de_pa" in df.columns else df.get("pa_ok", False)
        if "qtd_registros_de_peso_altura" in df.columns:
            df["c4_c_ok"] = parse_count(df["qtd_registros_de_peso_altura"]).fillna(0).ge(1)
        else:
            df["c4_c_ok"] = df.get("antropometria_ok", False)
        if "qtd_visitas_domiciliares" in df.columns:
            df["c4_d_ok"] = parse_count(df["qtd_visitas_domiciliares"]).fillna(0).ge(2)
        else:
            df["c4_d_ok"] = df.get("visita_ok", False)
        if "hemoglobina_glicada" in df.columns:
            df["c4_e_ok"] = to_bool(df["hemoglobina_glicada"])
        else:
            df["c4_e_ok"] = df.get("hba1c_ok", False)
        if "avaliacao_dos_pes" in df.columns:
            df["c4_f_ok"] = to_bool(df["avaliacao_dos_pes"])
        else:
            df["c4_f_ok"] = df.get("pes_ok", False)

    # C5
    if indicator_code == "C5":
        df["c5_a_ok"] = to_bool(df["consulta_medica_enfermagem"]) if "consulta_medica_enfermagem" in df.columns else df.get("consulta_ok", False)
        df["c5_b_ok"] = to_bool(df["afericao_de_pa"]) if "afericao_de_pa" in df.columns else df.get("pa_ok", False)
        if "qtd_registros_de_peso_altura" in df.columns:
            df["c5_c_ok"] = parse_count(df["qtd_registros_de_peso_altura"]).fillna(0).ge(1)
        else:
            df["c5_c_ok"] = df.get("antropometria_ok", False)
        if "qtd_visitas_domiciliares" in df.columns:
            df["c5_d_ok"] = parse_count(df["qtd_visitas_domiciliares"]).fillna(0).ge(2)
        else:
            df["c5_d_ok"] = df.get("visita_ok", False)

    # C6
    if indicator_code == "C6":
        if "consulta_medica_enfermagem" in df.columns:
            df["consulta_ok"] = to_bool(df["consulta_medica_enfermagem"])
        if "qtd_registros_de_peso_altura" in df.columns:
            df["antropometria_ok"] = parse_count(df["qtd_registros_de_peso_altura"]).fillna(0).ge(1)
        if "qtd_visitas_domiciliares" in df.columns:
            df["visitas_ok"] = parse_count(df["qtd_visitas_domiciliares"]).fillna(0).ge(2)
        if "vacina_influenza" in df.columns:
            df["influenza_ok"] = to_bool(df["vacina_influenza"])

    # C7
    if indicator_code == "C7":
        if "citopatologico_ou_hpv_ok" in df.columns:
            df["c7_a_ok"] = to_bool(df["citopatologico_ou_hpv_ok"])
        else:
            df["c7_a_ok"] = df.get("citopatologico_ok", False)
        if "vacina_hpv_ok" in df.columns:
            df["c7_b_ok"] = to_bool(df["vacina_hpv_ok"])
        if "saude_sexual_reprodutiva_ok" in df.columns:
            df["c7_c_ok"] = to_bool(df["saude_sexual_reprodutiva_ok"])
        if "mamografia_ok" in df.columns:
            df["c7_d_ok"] = to_bool(df["mamografia_ok"])

    return df

# =========================
# Cálculos
# =========================


def calculate_score_indicator(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    df = df.copy()
    weights = spec.weights or {}
    for c in list(weights.keys()):
        ensure_column(df, c, False)

    total_score = np.zeros(len(df), dtype=float)
    total_pendencias = np.zeros(len(df), dtype=int)

    for col, weight in weights.items():
        pratica_ok = to_bool(df[col])
        total_score += np.where(pratica_ok, weight, 0)
        total_pendencias += np.where(~pratica_ok, 1, 0)

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
    for col, peso in weights.items():
        if col not in df.columns:
            continue
        total = len(df)
        realizados = int(to_bool(df[col]).sum())
        nao_realizados = max(total - realizados, 0)
        perc = round((realizados / total) * 100, 1) if total else 0.0
        rows.append(
            {
                "Boa prática": label_boa_pratica(spec.code, col),
                "coluna": col,
                "Peso": peso,
                "Realizados": realizados,
                "% Realizado": perc,
                "Não realizado": nao_realizados,
            }
        )
    return pd.DataFrame(rows)

# =========================
# Filtros
# =========================


def apply_global_filters(df: pd.DataFrame, spec: IndicatorSpec) -> Tuple[pd.DataFrame, Optional[str]]:
    with st.sidebar:
        st.header("Filtros do painel")
        equipes = sorted([str(e) for e in df.get("equipe", pd.Series(dtype=str)).dropna().unique() if str(e).strip()])
        microareas = sorted([str(m) for m in df.get("micro_area", pd.Series(dtype=str)).dropna().unique() if str(m).strip()])
        faixas = sorted([str(f) for f in df.get("faixa_etaria", pd.Series(dtype=str)).dropna().unique() if str(f).strip()])

        eq_sel = st.multiselect("Por equipe", equipes)
        ma_sel = st.multiselect("Por microárea", microareas)
        fx_sel = st.multiselect("Por faixa etária", faixas)

        bp_df_full = build_good_practices_df(df, spec)
        pend_options = ["Todos"]
        label_to_col: Dict[str, str] = {}
        for _, row in bp_df_full.iterrows():
            label = row["Boa prática"]
            col = row["coluna"]
            pend_options.append(label)
            label_to_col[label] = col
        if "pendencias" in df.columns:
            pend_options.insert(1, "Sem pendências")

        pend_sel = st.selectbox("Por pendências", pend_options)

    out = df.copy()
    if eq_sel:
        out = out[out["equipe"].astype(str).isin(eq_sel)]
    if ma_sel:
        out = out[out["micro_area"].astype(str).isin(ma_sel)]
    if fx_sel:
        out = out[out["faixa_etaria"].astype(str).isin(fx_sel)]

    if pend_sel == "Sem pendências" and "pendencias" in out.columns:
        out = out[out["pendencias"] == 0]
    elif pend_sel != "Todos" and pend_sel in label_to_col:
        col = label_to_col[pend_sel]
        if col in out.columns:
            out = out[~to_bool(out[col])]

    selected_label = pend_sel if pend_sel != "Todos" else None
    return out, selected_label

# =========================
# Renderização
# =========================


def render_good_practices(df: pd.DataFrame, spec: IndicatorSpec):
    bp_df = build_good_practices_df(df, spec)
    st.markdown("### Cumprimento das boas práticas")
    if bp_df.empty:
        st.info("Não foi possível identificar boas práticas estruturadas para este relatório.")
        return

    bp_df_display = bp_df.copy()
    if "% Realizado" in bp_df_display.columns:
        bp_df_display["% Realizado"] = bp_df_display["% Realizado"].map(lambda v: f"{v:.1f}%" if pd.notna(v) else "")

    st.dataframe(
        bp_df_display[["Boa prática", "Peso", "Realizados", "% Realizado", "Não realizado"]],
        use_container_width=True,
    )


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
    desempenho = classificar_score(media_score)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de Pacientes", total)
    c2.metric("Score", f"{media_score:.1f}")
    c3.metric("Desempenho", desempenho)

    colg1, colg2 = st.columns(2)

    with colg1:
        if total > 0:
            bp_df = build_good_practices_df(df_scored, spec)
            if not bp_df.empty:
                bp_df = bp_df.copy()
                bp_df["Letra"] = bp_df["Boa prática"].str.extract(r"^([A-Z])", expand=False).fillna("")
                fig_bp = px.bar(
                    bp_df,
                    x="Letra",
                    y="% Realizado",
                    text="% Realizado",
                    title="Percentual de realização por boa prática (A–E)",
                )
                fig_bp.update_layout(xaxis_title="Boa prática", yaxis_title="%")
                st.plotly_chart(fig_bp, use_container_width=True)

    with colg2:
        class_df = df_scored["classificacao"].value_counts().reset_index()
        class_df.columns = ["Classificação", "Quantidade"]
        fig_class = px.pie(
            class_df,
            names="Classificação",
            values="Quantidade",
            title="Distribuição dos pacientes por faixa de desempenho",
        )
        st.plotly_chart(fig_class, use_container_width=True)

    render_good_practices(df_scored, spec)
    render_nominal(df_scored, spec)


def render_percentual_dashboard(df: pd.DataFrame, spec: IndicatorSpec):
    df_calc, indicador = calculate_percentual_indicator(df, spec)
    total = len(df_calc)
    desempenho = classificar_score(indicador)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de Pacientes", total)
    c2.metric("Score", f"{indicador:.1f}")
    c3.metric("Desempenho", desempenho)

    if "equipe" in df_calc.columns:
        by_team = (
            df_calc.groupby("equipe", dropna=False)
            .agg(numerador=("numerador", "sum"), denominador=("denominador", "sum"))
            .reset_index()
        )
        by_team["percentual"] = np.where(
            by_team["denominador"] > 0,
            by_team["numerador"] / by_team["denominador"] * 100,
            0,
        )
        st.dataframe(by_team, use_container_width=True)
        fig = px.bar(by_team, x="equipe", y="percentual", title="Indicador por equipe")
        st.plotly_chart(fig, use_container_width=True)

    render_nominal(df_calc, spec)


def render_nominal(df: pd.DataFrame, spec: IndicatorSpec):
    st.markdown("### Lista nominal")

    preferred_cols = [
        "nome",
        "cpf",
        "cns",
        "idade",
        "faixa_etaria",
        "endereco",
        "equipe",
        "micro_area",
        "equipe_vinculo",
        "tipo_equipe",
        "score",
        "classificacao",
        "pendencias",
        "cadastro_ok",
        "atendimento_ok",
        "consulta_ok",
        "pa_ok",
        "antropometria_ok",
        "visita_ok",
        "visitas_ok",
        "hba1c_ok",
        "pes_ok",
        "influenza_ok",
        "citopatologico_ok",
        "mamografia_ok",
        "exame_ok",
    ]

    cols = [c for c in preferred_cols if c in df.columns]
    if not cols:
        cols = list(df.columns)

    st.dataframe(df[cols], use_container_width=True, height=420)

    csv_bytes = df[cols].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Baixar CSV filtrado",
        data=csv_bytes,
        file_name=f"{spec.code.lower()}_lista_filtrada.csv",
        mime="text/csv",
    )

    st.download_button(
        "Baixar Excel filtrado",
        data=export_excel_bytes(df[cols]),
        file_name=f"{spec.code.lower()}_lista_filtrada.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# =========================
# Aplicação
# =========================


def main():
    st.title("Saúde 360 APS - Dashboard multipainel")
    st.caption(
        "Painel expandido para os 7 indicadores com leitura flexível de relatórios e cálculo operacional local."
    )

    st.sidebar.header("Importação")
    uploaded_file = st.sidebar.file_uploader(
        "Envie um relatório CSV/XLS/XLSX", type=["csv", "xls", "xlsx"]
    )

    st.sidebar.header("Indicador")
    manual_indicator = st.sidebar.selectbox(
        "Selecionar manualmente (opcional)",
        ["Automático"] + [f"{k} - {v.name}" for k, v in INDICATORS.items()],
    )

    if uploaded_file is None:
        st.info(
            "Envie um relatório para começar. O app tenta identificar o indicador automaticamente "
            "pelo nome do arquivo e pelas colunas."
        )
        st.stop()

    try:
        df_raw = read_uploaded_file(uploaded_file)
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {e}")
        st.stop()

    detected = detect_indicator_from_columns(
        pd.DataFrame(columns=[normalize_col(c) for c in df_raw.columns]),
        uploaded_file.name,
    )
    selected_code = (
        manual_indicator.split(" ")[0]
        if manual_indicator != "Automático"
        else detected
    )

    if selected_code is None:
        st.warning(
            "Não foi possível identificar automaticamente o indicador. "
            "Escolha manualmente na barra lateral."
        )
        st.stop()

    spec = INDICATORS[selected_code]
    df = preprocess_df(df_raw, selected_code)

    df_filtered, pend_label = apply_global_filters(df, spec)

    st.success(f"Indicador em análise: {spec.code} - {spec.name}")
    if pend_label:
        st.caption(f"Filtro de pendências aplicado: {pend_label}")

    st.markdown(f"## {spec.code} - {spec.name}")
    st.write(spec.description)
    if spec.type == "score":
        render_score_dashboard(df_filtered, spec)
    else:
        render_percentual_dashboard(df_filtered, spec)


if __name__ == "__main__":
    main()
