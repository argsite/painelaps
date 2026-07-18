import io
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Saúde 360 - Indicadores C2 a C7", layout="wide")

INDICADORES = {
    "C2": "Cuidado no Desenvolvimento Infantil",
    "C3": "Cuidado na Gestação e Puerpério",
    "C4": "Cuidado da Pessoa com Diabetes",
    "C5": "Cuidado da Pessoa com Hipertensão",
    "C6": "Cuidado da Pessoa Idosa",
    "C7": "Cuidado da Mulher na Prevenção do Câncer",
}

PESOS = {
    "C2": {
        "consulta_1_mes_ok": 20,
        "consultas_ok": 20,
        "antropometria_ok": 20,
        "visitas_ok": 20,
        "vacinal_ok": 20,
    },
    "C3": {
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
    "C4": {
        "consulta_ok": 20,
        "pa_ok": 15,
        "antropometria_ok": 15,
        "visitas_ok": 20,
        "hba1c_ok": 15,
        "pes_ok": 15,
    },
    "C5": {
        "consulta_ok": 25,
        "pa_ok": 25,
        "antropometria_ok": 25,
        "visitas_ok": 25,
    },
    "C6": {
        "consulta_ok": 25,
        "antropometria_ok": 25,
        "visitas_ok": 25,
        "influenza_ok": 25,
    },
    "C7": {
        "colo_utero_ok": 20,
        "hpv_ok": 30,
        "saude_reprodutiva_ok": 30,
        "mama_ok": 20,
    },
}

BASE_COLS = [
    "nome", "cpf", "cns", "data_nascimento", "idade", "endereco", "equipe", "microarea",
    "equipe_vinculo", "cadastro_atualizado", "data_atualizacao_cadastro", "acompanhado"
]


def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_flag(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().upper()
    if s in {"S", "SIM", "TRUE", "1", "SIM, PARCIAL", "SIM PARCIAL"}:
        return True
    if s in {"N", "NAO", "NÃO", "FALSE", "0"}:
        return False
    if s == "-":
        return np.nan
    return np.nan


def parse_count(x):
    if pd.isna(x):
        return 0
    s = str(x).strip().replace(",", ".")
    s = s.replace("+", "")
    m = re.search(r"\d+(?:\.\d+)?", s)
    if not m:
        return 0
    try:
        return float(m.group())
    except Exception:
        return 0


def classify_score(score):
    if score >= 75:
        return "Ótimo"
    if score >= 50:
        return "Bom"
    if score >= 25:
        return "Suficiente"
    return "Regular"


def smart_read(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    try:
        return pd.read_excel(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        data = uploaded_file.read()
        bio = io.BytesIO(data)
        return pd.read_html(bio)[0]


def ensure_columns(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def standardize_base(df, mapping):
    df = df.rename(columns=mapping).copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = ensure_columns(df, BASE_COLS)
    return df


def finalize_df(df, indicador, practice_cols, applicable_cols=None):
    applicable_cols = applicable_cols or {}
    weights = PESOS[indicador]

    for col in practice_cols:
        if col not in df.columns:
            df[col] = False
        df[col] = df[col].fillna(False).astype(bool)

    score = np.zeros(len(df), dtype=float)
    pend = np.zeros(len(df), dtype=int)
    aplic = np.zeros(len(df), dtype=int)

    for col in practice_cols:
        peso = weights[col]
        app_col = applicable_cols.get(col)
        if app_col and app_col in df.columns:
            app = df[app_col].fillna(False).astype(bool)
        else:
            app = pd.Series(True, index=df.index)
        score += np.where(app & df[col], peso, 0)
        pend += np.where(app & (~df[col]), 1, 0)
        aplic += np.where(app, 1, 0)

    df["score"] = score
    df["pendencias"] = pend
    df["praticas_aplicaveis"] = aplic
    df["classificacao"] = df["score"].apply(classify_score)
    df["indicador"] = indicador
    df["nome_indicador"] = INDICADORES[indicador]
    return df


def processar_c2(df):
    mapping = {
        "Nome Completo": "nome",
        "Nome da Mãe": "nome_mae",
        "CPF": "cpf",
        "CNS": "cns",
        "Data Nascimento": "data_nascimento",
        "Idade": "idade",
        "Endereço": "endereco",
        "Equipe Área": "equipe",
        "Microárea": "microarea",
        "Equipe Vínculo": "equipe_vinculo",
        "Cadastro Atualizado": "cadastro_atualizado",
        "Data Atualização Cadastro": "data_atualizacao_cadastro",
        "Consulta médica/enfermagem 1º mês": "consulta_1_mes",
        "Nr. Consultas": "nr_consultas",
        "Qtd. Registros de peso/altura": "qtd_antropometria",
        "Visita Domiciliar 1º mês": "visita_1_mes",
        "Visita Domiciliar 6º mês": "visita_6_mes",
        "Válida no quadrimestre": "valida_quadrimestre",
        "Esquema vacinal completo": "esquema_vacinal_completo",
        "Vacina Pentavalente": "vacina_pentavalente",
        "Vacina Pólio Injetável": "vacina_polio",
        "Vacina Sarampo, Caxumba e Rubéola": "vacina_scr",
        "Vacina Pneumocócica": "vacina_pneumo",
        "Acompanhado": "acompanhado",
    }
    df = standardize_base(df, mapping)
    df["consulta_1_mes_ok"] = df["consulta_1_mes"].apply(parse_flag).fillna(False)
    df["consultas_ok"] = df["nr_consultas"].apply(parse_count) >= 9
    df["antropometria_ok"] = df["qtd_antropometria"].apply(parse_count) >= 9
    df["visitas_ok"] = df["visita_1_mes"].apply(parse_flag).fillna(False) & df["visita_6_mes"].apply(parse_flag).fillna(False)
    df["vacinal_ok"] = df["esquema_vacinal_completo"].apply(parse_flag).fillna(False)
    practice_cols = ["consulta_1_mes_ok", "consultas_ok", "antropometria_ok", "visitas_ok", "vacinal_ok"]
    return finalize_df(df, "C2", practice_cols)


def processar_c3(df):
    mapping = {
        "Nome Completo": "nome",
        "CNS": "cns",
        "CPF": "cpf",
        "Data Nascimento": "data_nascimento",
        "Idade": "idade",
        "Endereço": "endereco",
        "Equipe Área": "equipe",
        "Microárea": "microarea",
        "Equipe Vínculo": "equipe_vinculo",
        "Cadastro Atualizado": "cadastro_atualizado",
        "Data Atualização Cadastro": "data_atualizacao_cadastro",
        "Consulta de pré-natal até 12 semanas": "pre_natal_12s",
        "Consulta médica/enfermagem Gestação": "consultas_gest",
        "Aferição de pressão arterial": "afericoes_pa",
        "Registro de peso/altura": "registro_peso_altura",
        "Válida no quadrimestre": "valida_quadrimestre",
        "Visitas domiciliares (ACS/TACS) Gestação": "visitas_gest",
        "Vacina dTpa": "dtpa",
        "Teste Rápido sífilis primeiro trimestre": "tri1_sifilis",
        "Teste Rápido HIV primeiro trimestre": "tri1_hiv",
        "Teste Rápido Hepatite B primeiro trimestre": "tri1_hbv",
        "Teste Rápido Hepatite C primeiro trimestre": "tri1_hcv",
        "Teste Rápido sífilis terceiro trimestre": "tri3_sifilis",
        "Teste Rápido HIV terceiro trimestre": "tri3_hiv",
        "Consulta médica/enfermagem Puerpério": "consulta_puerperio",
        "Visitas domiciliares (ACS/TACS) Puerpério": "visita_puerperio",
        "Avaliação odontológica Gestação": "avaliacao_odonto",
        "Acompanhado": "acompanhado",
    }
    df = standardize_base(df, mapping)
    df["pre_natal_12s_ok"] = df["pre_natal_12s"].apply(parse_flag).fillna(False)
    df["consultas_gest_ok"] = df["consultas_gest"].apply(parse_count) >= 7
    df["pa_ok"] = df["afericoes_pa"].apply(parse_count) >= 7
    df["antropometria_ok"] = df["registro_peso_altura"].apply(parse_count) >= 7
    df["visitas_gest_ok"] = df["visitas_gest"].apply(parse_count) >= 3
    df["dtpa_ok"] = df["dtpa"].apply(parse_flag).fillna(False)
    df["tri1_ok"] = (
        df["tri1_sifilis"].apply(parse_flag).fillna(False)
        & df["tri1_hiv"].apply(parse_flag).fillna(False)
        & df["tri1_hbv"].apply(parse_flag).fillna(False)
        & df["tri1_hcv"].apply(parse_flag).fillna(False)
    )
    df["tri3_ok"] = (
        df["tri3_sifilis"].apply(parse_flag).fillna(False)
        & df["tri3_hiv"].apply(parse_flag).fillna(False)
    )
    df["puerperio_consulta_ok"] = df["consulta_puerperio"].apply(parse_flag).fillna(False)
    df["puerperio_visita_ok"] = df["visita_puerperio"].apply(parse_flag).fillna(False)
    df["odonto_ok"] = df["avaliacao_odonto"].apply(parse_flag).fillna(False)
    practice_cols = [
        "pre_natal_12s_ok", "consultas_gest_ok", "pa_ok", "antropometria_ok", "visitas_gest_ok",
        "dtpa_ok", "tri1_ok", "tri3_ok", "puerperio_consulta_ok", "puerperio_visita_ok", "odonto_ok"
    ]
    return finalize_df(df, "C3", practice_cols)


def processar_c4(df):
    mapping = {
        "Nome Completo": "nome",
        "CPF": "cpf",
        "CNS": "cns",
        "Data Nascimento": "data_nascimento",
        "Idade": "idade",
        "Endereço": "endereco",
        "Equipe Área": "equipe",
        "Microárea": "microarea",
        "Equipe Vínculo": "equipe_vinculo",
        "Cadastro Atualizado": "cadastro_atualizado",
        "Data Atualização Cadastro": "data_atualizacao_cadastro",
        "Consulta Médica/Enfermagem": "consulta",
        "Aferição de PA": "pa",
        "Qtd. Registros de peso/altura": "antropometria_qtd",
        "Qtd. Visitas Domiciliares": "visitas_qtd",
        "Hemoglobina Glicada": "hba1c",
        "Avaliação dos pés": "pes",
        "Acompanhado": "acompanhado",
    }
    df = standardize_base(df, mapping)
    df["consulta_ok"] = df["consulta"].apply(parse_flag).fillna(False)
    df["pa_ok"] = df["pa"].apply(parse_flag).fillna(False)
    df["antropometria_ok"] = df["antropometria_qtd"].apply(parse_count) >= 1
    df["visitas_ok"] = df["visitas_qtd"].apply(parse_count) >= 2
    df["hba1c_ok"] = df["hba1c"].apply(parse_flag).fillna(False)
    df["pes_ok"] = df["pes"].apply(parse_flag).fillna(False)
    practice_cols = ["consulta_ok", "pa_ok", "antropometria_ok", "visitas_ok", "hba1c_ok", "pes_ok"]
    return finalize_df(df, "C4", practice_cols)


def processar_c5(df):
    mapping = {
        "Nome Completo": "nome",
        "CPF": "cpf",
        "CNS": "cns",
        "Data Nascimento": "data_nascimento",
        "Idade": "idade",
        "Endereço": "endereco",
        "Equipe Área": "equipe",
        "Microárea": "microarea",
        "Equipe Vínculo": "equipe_vinculo",
        "Cadastro Atualizado": "cadastro_atualizado",
        "Data Atualização Cadastro": "data_atualizacao_cadastro",
        "Consulta Médica/Enfermagem": "consulta",
        "Qtd. Registros de peso/altura": "antropometria_qtd",
        "Qtd. Visitas Domiciliares": "visitas_qtd",
        "Aferição de pressão arterial": "pa",
        "Acompanhado": "acompanhado",
    }
    df = standardize_base(df, mapping)
    df["consulta_ok"] = df["consulta"].apply(parse_flag).fillna(False)
    df["pa_ok"] = df["pa"].apply(parse_flag).fillna(False)
    df["antropometria_ok"] = df["antropometria_qtd"].apply(parse_count) >= 1
    df["visitas_ok"] = df["visitas_qtd"].apply(parse_count) >= 2
    practice_cols = ["consulta_ok", "pa_ok", "antropometria_ok", "visitas_ok"]
    return finalize_df(df, "C5", practice_cols)


def processar_c6(df):
    mapping = {
        "Nome Completo": "nome",
        "CPF": "cpf",
        "CNS": "cns",
        "Data Nascimento": "data_nascimento",
        "Idade": "idade",
        "Endereço": "endereco",
        "Equipe Área": "equipe",
        "Microárea": "microarea",
        "Equipe Vínculo": "equipe_vinculo",
        "Cadastro Atualizado": "cadastro_atualizado",
        "Data Atualização Cadastro": "data_atualizacao_cadastro",
        "Consulta Médica/Enfermagem": "consulta",
        "Qtd. Registros de peso/altura": "antropometria_qtd",
        "Qtd. Visitas Domiciliares": "visitas_qtd",
        "Vacina influenza": "influenza",
        "Acompanhado": "acompanhado",
    }
    df = standardize_base(df, mapping)
    df["consulta_ok"] = df["consulta"].apply(parse_flag).fillna(False)
    df["antropometria_ok"] = df["antropometria_qtd"].apply(parse_count) >= 1
    df["visitas_ok"] = df["visitas_qtd"].apply(parse_count) >= 2
    df["influenza_ok"] = df["influenza"].apply(parse_flag).fillna(False)
    practice_cols = ["consulta_ok", "antropometria_ok", "visitas_ok", "influenza_ok"]
    return finalize_df(df, "C6", practice_cols)


def processar_c7(df):
    mapping = {
        "Nome Completo": "nome",
        "CPF": "cpf",
        "CNS": "cns",
        "Data Nascimento": "data_nascimento",
        "Idade": "idade",
        "Endereço": "endereco",
        "Equipe Área": "equipe",
        "Microárea": "microarea",
        "Equipe Vínculo": "equipe_vinculo",
        "Cadastro Atualizado": "cadastro_atualizado",
        "Data Atualização Cadastro": "data_atualizacao_cadastro",
        "Rast. Câncer do Colo do Útero": "colo_utero",
        "Vacina HPV entre 9 e 14 anos": "hpv",
        "Atend. Saúde Reprodutiva": "saude_reprodutiva",
        "Rast. Câncer de mama": "mama",
        "Acompanhado": "acompanhado",
    }
    df = standardize_base(df, mapping)
    for src in ["colo_utero", "hpv", "saude_reprodutiva", "mama"]:
        df[f"{src}_aplicavel"] = df[src].astype(str).str.strip().ne("-")
        df[f"{src}_ok"] = df[src].apply(parse_flag).fillna(False)
    practice_cols = ["colo_utero_ok", "hpv_ok", "saude_reprodutiva_ok", "mama_ok"]
    applicable_cols = {
        "colo_utero_ok": "colo_utero_aplicavel",
        "hpv_ok": "hpv_aplicavel",
        "saude_reprodutiva_ok": "saude_reprodutiva_aplicavel",
        "mama_ok": "mama_aplicavel",
    }
    return finalize_df(df, "C7", practice_cols, applicable_cols)


PROCESSADORES = {
    "C2": processar_c2,
    "C3": processar_c3,
    "C4": processar_c4,
    "C5": processar_c5,
    "C6": processar_c6,
    "C7": processar_c7,
}

PRACTICE_LABELS = {
    "C2": {
        "consulta_1_mes_ok": "Consulta até 30 dias",
        "consultas_ok": "9 consultas",
        "antropometria_ok": "9 registros de peso/altura",
        "visitas_ok": "2 visitas (1º e 6º mês)",
        "vacinal_ok": "Esquema vacinal completo",
    },
    "C3": {
        "pre_natal_12s_ok": "Pré-natal até 12 semanas",
        "consultas_gest_ok": "7 consultas na gestação",
        "pa_ok": "7 aferições de PA",
        "antropometria_ok": "7 registros de peso/altura",
        "visitas_gest_ok": "3 visitas na gestação",
        "dtpa_ok": "Vacina dTpa",
        "tri1_ok": "Testes 1º trimestre",
        "tri3_ok": "Testes 3º trimestre",
        "puerperio_consulta_ok": "Consulta no puerpério",
        "puerperio_visita_ok": "Visita no puerpério",
        "odonto_ok": "Avaliação odontológica",
    },
    "C4": {
        "consulta_ok": "Consulta médica/enfermagem",
        "pa_ok": "Aferição de PA",
        "antropometria_ok": "Peso/altura",
        "visitas_ok": "2 visitas domiciliares",
        "hba1c_ok": "Hemoglobina glicada",
        "pes_ok": "Avaliação dos pés",
    },
    "C5": {
        "consulta_ok": "Consulta médica/enfermagem",
        "pa_ok": "Aferição de PA",
        "antropometria_ok": "Peso/altura",
        "visitas_ok": "2 visitas domiciliares",
    },
    "C6": {
        "consulta_ok": "Consulta médica/enfermagem",
        "antropometria_ok": "Peso/altura",
        "visitas_ok": "2 visitas domiciliares",
        "influenza_ok": "Vacina influenza",
    },
    "C7": {
        "colo_utero_ok": "Rastreio colo do útero",
        "hpv_ok": "Vacina HPV",
        "saude_reprodutiva_ok": "Atendimento saúde reprodutiva",
        "mama_ok": "Rastreio câncer de mama",
    },
}


st.title("Saúde 360 - Dashboard dos Indicadores C2 a C7")
st.caption("Leitura de relatórios nominais, cálculo de score por pessoa, pendências, consolidados por equipe e listas operacionais.")

with st.sidebar:
    st.header("Upload")
    indicador = st.selectbox("Indicador", list(INDICADORES.keys()), format_func=lambda x: f"{x} - {INDICADORES[x]}")
    arquivo = st.file_uploader("Selecione o relatório exportado", type=["xls", "xlsx", "csv"])
    st.markdown("---")
    st.markdown("**Regras implementadas**")
    st.write("- C2: 5 boas práticas de 20 pontos.")
    st.write("- C3: 11 boas práticas com pesos 10 + 10x9.")
    st.write("- C4: 20, 15, 15, 20, 15, 15.")
    st.write("- C5: 4 práticas de 25 pontos.")
    st.write("- C6: 4 práticas de 25 pontos.")
    st.write("- C7: 20, 30, 30, 20; '-' tratado como não aplicável.")

if not arquivo:
    st.info("Faça upload de um relatório para gerar o painel.")
    st.stop()

try:
    bruto = smart_read(arquivo)
    processado = PROCESSADORES[indicador](bruto)
except Exception as e:
    st.error(f"Erro ao ler/processar o arquivo: {e}")
    st.stop()

processado["equipe"] = processado["equipe"].astype(str)
processado["microarea"] = processado["microarea"].astype(str)
processado["acompanhado_flag"] = processado["acompanhado"].apply(parse_flag)

with st.sidebar:
    equipes = sorted([e for e in processado["equipe"].dropna().astype(str).unique().tolist() if e and e != 'nan'])
    equipe_sel = st.multiselect("Equipe", equipes, default=equipes)
    micro_sel = st.multiselect(
        "Microárea",
        sorted([m for m in processado["microarea"].dropna().astype(str).unique().tolist() if m and m != 'nan']),
        default=sorted([m for m in processado["microarea"].dropna().astype(str).unique().tolist() if m and m != 'nan'])
    )
    somente_pendentes = st.checkbox("Mostrar apenas com pendências")
    somente_nao_acompanhados = st.checkbox("Mostrar apenas não acompanhados")
    busca = st.text_input("Buscar nome / CPF / CNS")

f = processado.copy()
if equipe_sel:
    f = f[f["equipe"].astype(str).isin(equipe_sel)]
if micro_sel:
    f = f[f["microarea"].astype(str).isin(micro_sel)]
if somente_pendentes:
    f = f[f["pendencias"] > 0]
if somente_nao_acompanhados:
    f = f[f["acompanhado_flag"].fillna(False) == False]
if busca:
    b = busca.strip().lower()
    f = f[
        f["nome"].astype(str).str.lower().str.contains(b, na=False)
        | f["cpf"].astype(str).str.contains(b, na=False)
        | f["cns"].astype(str).str.contains(b, na=False)
    ]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Pessoas no painel", len(f))
c2.metric("Score médio", f["score"].mean().round(1) if len(f) else 0)
c3.metric("Acompanhados", int(f["acompanhado_flag"].fillna(False).sum()) if len(f) else 0)
c4.metric("Com pendências", int((f["pendencias"] > 0).sum()) if len(f) else 0)

agg = (
    f.groupby("equipe", dropna=False)
    .agg(
        pessoas=("nome", "count"),
        score_medio=("score", "mean"),
        acompanhados=("acompanhado_flag", lambda s: int(pd.Series(s).fillna(False).sum())),
        pendentes=("pendencias", lambda s: int((pd.Series(s) > 0).sum())),
    )
    .reset_index()
)
agg["score_medio"] = agg["score_medio"].round(1)

st.subheader("Desempenho por equipe")
if len(agg):
    fig = px.bar(agg.sort_values("score_medio", ascending=False), x="equipe", y="score_medio", color="score_medio", text="score_medio")
    fig.update_layout(height=360, coloraxis_showscale=False, xaxis_title="Equipe", yaxis_title="Score médio")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Sem dados após os filtros.")

practice_map = PRACTICE_LABELS[indicador]
practice_cols = list(practice_map.keys())
practice_summary = []
for col in practice_cols:
    app_col = col.replace("_ok", "_aplicavel")
    if app_col in f.columns:
        base = f[f[app_col].fillna(False)]
    else:
        base = f
    total = len(base)
    ok = int(base[col].fillna(False).sum()) if total else 0
    pct = round((ok / total) * 100, 1) if total else 0
    practice_summary.append({"Boa prática": practice_map[col], "Realizaram": ok, "Elegíveis": total, "%": pct})
practice_df = pd.DataFrame(practice_summary)

col_a, col_b = st.columns([1.1, 1])
with col_a:
    st.subheader("Cumprimento por boa prática")
    st.dataframe(practice_df, use_container_width=True, hide_index=True)
with col_b:
    fig2 = px.bar(practice_df.sort_values("%"), x="%", y="Boa prática", orientation="h", text="%")
    fig2.update_layout(height=max(320, 70 * len(practice_df)), yaxis_title="", xaxis_title="% de cumprimento")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Lista nominal")
nominal = f.copy()
for col, label in practice_map.items():
    nominal[label] = np.where(nominal[col].fillna(False), "OK", "Pendente")
    app_col = col.replace("_ok", "_aplicavel")
    if app_col in nominal.columns:
        nominal[label] = np.where(nominal[app_col].fillna(False), nominal[label], "N/A")

cols_show = [
    c for c in [
        "nome", "cpf", "cns", "idade", "equipe", "microarea", "equipe_vinculo", "score", "pendencias", "classificacao", "acompanhado"
    ] if c in nominal.columns
] + list(practice_map.values())

st.dataframe(
    nominal[cols_show].sort_values(["pendencias", "score", "nome"], ascending=[False, True, True]),
    use_container_width=True,
    hide_index=True,
)

st.subheader("Pendências operacionais")
pend_ops = []
for _, row in f.iterrows():
    faltantes = []
    for col, label in practice_map.items():
        app_col = col.replace("_ok", "_aplicavel")
        aplicavel = bool(row[app_col]) if app_col in f.columns and not pd.isna(row[app_col]) else True
        if aplicavel and not bool(row[col]):
            faltantes.append(label)
    pend_ops.append(
        {
            "nome": row.get("nome"),
            "cpf": row.get("cpf"),
            "cns": row.get("cns"),
            "equipe": row.get("equipe"),
            "microarea": row.get("microarea"),
            "score": row.get("score"),
            "pendencias": row.get("pendencias"),
            "faltantes": "; ".join(faltantes),
        }
    )
pend_df = pd.DataFrame(pend_ops)
pend_df = pend_df[pend_df["pendencias"] > 0].sort_values(["pendencias", "score", "nome"], ascending=[False, True, True])
st.dataframe(pend_df, use_container_width=True, hide_index=True)

csv = pend_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("Baixar CSV de pendências", data=csv, file_name=f"pendencias_{indicador.lower()}.csv", mime="text/csv")

with st.expander("Pré-visualização das colunas do arquivo lido"):
    st.write(list(bruto.columns))
    st.dataframe(bruto.head(20), use_container_width=True)
