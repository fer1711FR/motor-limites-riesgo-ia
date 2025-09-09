import os, joblib, yaml
import pandas as pd
import streamlit as st
from pathlib import Path
from preprocessing import preprocess_input
from openai import OpenAI
from dotenv import load_dotenv

# === CONFIGURACIÓN OPENAI ===
load_dotenv()
#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=api_key)

# === CARGA DE CONFIGURACIÓN ===
CFG = yaml.safe_load(open("configs/config.yaml"))
ART_DIR = Path("artifacts")

# === CARGA DEL MODELO ===
@st.cache_resource
def load_bundles():
    pd_model = joblib.load(ART_DIR / "comportamiento_xgb_VII_vf.pkl")
    cl_bundle = joblib.load(ART_DIR / "cluster_model.pkl")
    return pd_model, cl_bundle

# === FUNCIONES AUXILIARES ===
def annuity_factor(i_m, n):
    return (i_m * (1 + i_m) ** n) / ((1 + i_m) ** n - 1)

def bandas_multiplier(bandas_cfg: dict, pd: float) -> float:
    for name, rango in bandas_cfg.items():
        min_str, max_str = rango.split("-")
        min_val = float(min_str)
        max_val = float(max_str)
        if min_val <= pd <= max_val:
            if name == "low":
                return 1.0
            elif name == "medium":
                return 0.7
            elif name == "high":
                return 0.4
    return 0.0

bands_cfg = CFG["bandas_pd"]
bandas = []
for name, rng in bands_cfg.items():
    lo, hi = map(float, rng.split("-"))
    bandas.append({"name": name, "min_pd": lo, "max_pd": hi})
bandas = sorted(bandas, key=lambda x: x["max_pd"])

def get_banda(pdv: float) -> str:
    for b in bandas:
        if pdv >= b["min_pd"] and pdv <= b["max_pd"]:
            return b["name"]
    return f">{bandas[-1]['max_pd']}"

def build_explanation_llm(audiencia, pd, banda, cluster, limit_aff, limit_risk, limit_final, currency, top_factors):
    if audiencia == "cliente":
        prompt = f"""
        Eres un asistente financiero que explica evaluaciones de riesgo crediticio a un cliente.

        Explica en lenguaje sencillo y tranquilizador:
        - Probabilidad de incumplimiento: {pd:.2%}
        - Nivel de riesgo: {banda}
        - Cluster asignado: {cluster}
        - Límite calculado por capacidad de pago: {limit_aff:,.0f} {currency}
        - Límite calculado por riesgo: {limit_risk:,.0f} {currency}
        - Límite final recomendado: {limit_final:,.0f} {currency}
        - Factores más influyentes: {', '.join(top_factors)}
        """
    else:  # analista
        prompt = f"""
        Eres un asistente financiero que explica evaluaciones de riesgo crediticio a un analista técnico.

        Proporciona una explicación detallada y técnica, con detalles del cálculo y justificación
        - Probabilidad de incumplimiento: {pd:.2%}
        - Nivel de riesgo: {banda}
        - Cluster asignado: {cluster}
        - Límite calculado por capacidad de pago: {limit_aff:,.0f} {currency}
        - Límite calculado por riesgo: {limit_risk:,.0f} {currency}
        - Límite final recomendado: {limit_final:,.0f} {currency}
        - Factores más influyentes: {', '.join(top_factors)}
        """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # o "gpt-5"
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error en generación con LLM: {e}]"

def quantize(value, step):
    import math
    return math.floor(value / step) * step

def compute_limit(row_dict, cluster_id, pdv):
    cfg = CFG
    currency = cfg.get("currency","PEN")
    cparams = cfg["clusters"][cluster_id]
    pti_max = float(cparams["pti_max"])
    cap_base = float(cparams["cap_base"])
    max_limit = float(cparams["max_limit"])

    i_m = float(cfg["interest_rate_anual"])/12.0
    n = int(cfg["tenor_meses"])
    AF = annuity_factor(i_m, n)
    pago_max = max(0.0, row_dict["ingreso"] * pti_max - row_dict["pago_men_deuda"])
    limit_aff = pago_max / AF

    mult = bandas_multiplier(cfg["bandas_pd"], pdv)
    limit_risk = cap_base * mult * (1 - pdv) ** float(cfg["gamma_riesgo"])

    step = float(cfg["step_cuantizacion"])
    min_limit = float(cfg["min_limit"])
    limit = max(min(quantize(min(limit_aff, limit_risk), step), max_limit), 0.0)
    if 0 < limit < min_limit:
        limit = 0.0

    return {
        "limit_final": limit,
        "limit_aff": limit_aff,
        "limit_risk": limit_risk,
        "mult_banda": mult,
        "pago_max": pago_max,
        "AF": AF,
        "currency": currency,
        "pti_max": pti_max,
        "cap_base": cap_base
    }

# === NUEVA FUNCIÓN AUXILIAR ===
def prepare_for_cluster(df_in, pd_model, cl_bundle):
    # Preprocesa
    X_proc = preprocess_input(df_in)

    # Calcula probabilidad de default
    pdv = pd_model.predict_proba(X_proc)[:, 1]

    # Asegura que prob_default esté en el DataFrame
    X_proc = X_proc.copy()
    X_proc["prob_default"] = pdv

    # 🔑 Reordena columnas para que coincidan con el scaler
    feats_cl = cl_bundle["features"]
    X_proc = X_proc.reindex(columns=feats_cl, fill_value=0)

    return X_proc, pdv

# === STREAMLIT APP ===
st.set_page_config(page_title="Motor de Límites", layout="wide")
st.title("Motor Inteligente de Límite de Oferta")

with st.sidebar:
    st.header("Opciones")
    audiencia = st.selectbox("Audiencia", ["cliente", "analista"])

pd_model, cl_bundle = load_bundles()

tab1, tab2 = st.tabs(["Consulta individual", "Carga masiva"])

with tab1:
    st.subheader("Consulta individual")

    # === Cargar base de datos ===
    df_base = pd.read_csv("data/sample/base_var_comportamiento.csv")

    # Seleccionar cliente por índice
    cliente_idx = st.selectbox("Selecciona cliente (índice de fila)", df_base.index.tolist())

    # Extraer cliente
    df_in = df_base.iloc[[cliente_idx]].copy()

    # Permitir modificar ingreso y deuda manualmente
    cols = st.columns(2)
    ingreso = cols[0].number_input(
        "Ingreso mensual",
        min_value=0.0,
        value=float(df_in["ingreso"].iloc[0]) if "ingreso" in df_in else 3000.0,
        step=100.0
    )
    pago_men_deuda = cols[1].number_input(
        "Deuda mensual",
        min_value=0.0,
        value=float(df_in["pago_men_deuda"].iloc[0]) if "pago_men_deuda" in df_in else 500.0,
        step=50.0
    )

    # Actualizar valores en el DataFrame con los modificados
    df_in["ingreso"] = ingreso
    df_in["pago_men_deuda"] = pago_men_deuda

    
    X_proc_cl, pdv = prepare_for_cluster(df_in, pd_model, cl_bundle)

    feats_cl = cl_bundle["features"]
    X_proc_cl = X_proc_cl.reindex(columns=feats_cl, fill_value=0)

    X_scaled = cl_bundle["scaler"].transform(X_proc_cl)
    clid = int(cl_bundle["kmeans"].predict(X_scaled)[0])

    comp = compute_limit(df_in.iloc[0].to_dict(), clid, float(pdv[0]))

    st.metric("PD (impago)", f"{pdv[0]:.2%}")
    st.metric("Cluster", f"{clid}")
    st.metric("Límite recomendado", f"{comp['limit_final']:,.0f} {comp['currency']}")

    banda_txt = get_banda(float(pdv[0]))
    exp = build_explanation_llm(audiencia, float(pdv[0]), banda_txt, clid,
                                comp["limit_aff"], comp["limit_risk"], comp["limit_final"],
                                comp["currency"], ["ingreso", "pago_men_deuda"])  # 👈 usamos LLM
    st.subheader("Explicación Generativa")
    st.write(exp)

with tab2:
    st.subheader("Carga masiva (CSV)")
    up = st.file_uploader("Sube un CSV con las columnas de features", type=["csv"])
    if up is not None:
        df_in = pd.read_csv(up)

        X_proc, pdv = prepare_for_cluster(df_in, pd_model, cl_bundle)
        clid = cl_bundle["kmeans"].predict(cl_bundle["scaler"].transform(X_proc))

        out_rows = []
        for i, row in df_in.iterrows():
            comp = compute_limit(row.to_dict(), int(clid[i]), float(pdv[i]))
            out_rows.append({
                "pd": float(pdv[i]),
                "cluster": int(clid[i]),
                "limit_recommended": float(comp["limit_final"]),
                "limit_aff": float(comp["limit_aff"]),
                "limit_risk": float(comp["limit_risk"]),
                "currency": comp["currency"]
            })

        out = pd.concat([df_in.reset_index(drop=True), pd.DataFrame(out_rows)], axis=1)

        # === Mostrar tabla con columnas base + seleccionables ===
        base_cols = ["ingreso", "pago_men_deuda", "pd", "cluster", "limit_recommended", "limit_aff", "limit_risk"]  #
        extra_cols = st.multiselect(
            "Agregar columnas opcionales",
            [c for c in out.columns if c not in base_cols]
        )
        st.dataframe(out[base_cols + extra_cols].head(50))

        st.download_button("Descargar resultados", data=out.to_csv(index=False).encode("utf-8"),
                           file_name="resultados_limites.csv", mime="text/csv")

        idx = st.number_input("Selecciona índice de registro para explicación",
                              min_value=0, max_value=len(out)-1, step=1)
        row = out.iloc[idx]
        banda_txt = get_banda(row["pd"])
        exp = build_explanation_llm(audiencia, row["pd"], banda_txt, row["cluster"],
                                    row["limit_aff"], row["limit_risk"], row["limit_recommended"],
                                    row["currency"], ["ingreso", "pago_men_deuda"])  # 👈 usamos LLM
        st.subheader("Explicación Generativa")
        st.write(exp)