import pandas as pd
import joblib

expected_features = [
 "max_max_dias_atraso_deuda_directa_3um",
 "max_max_dias_atraso_deuda_directa_6um",
 "max_max_dias_atraso_deuda_directa_consumo_3um",
 "Ratio_utilizacion_tc_consumo",
 "prom_Ratio_utilizacion_tc_consumo_3um",
 "mto_lineadisp_tc_consumo_toh",
 "max_max_dias_atraso_deuda_directa_9um",
 "mto_lineadisp_tc_consumo",
 "max_max_dias_atraso_deuda_directa_consumo_6um",
 "max_max_dias_atraso_deuda_directa_consumo_toh_6um",
 "util_ul_mes",
 "max_max_dias_atraso_deuda_directa_toh_6um",
 "prom_Ratio_utilizacion_tc_consumo_toh_3um",
 "flag_100normal_toh_12um",
 "max_max_dias_atraso_deuda_directa_12um",
 "cnt_utl_meses_aldia_12m",
 "max_max_dias_atraso_deuda_directa_toh_9um",
 "prom_mto_linea_tc_consumo_toh_6um",
 "Ratio_utilizacion_tc_disponibilidad_efectivo",
 "meses_mora_360",
 "flag_100normal_ripley_6um",
 "max_max_dias_atraso_deuda_directa_consumo_toh_12um",
 "Ratio_utilizacion_tc_disponibilidad_efectivo_toh",
 "mto_deuda_directa_consumo_toh",
 "mto_deuda_directa_tc_disponibilidad_efectivo_revolvente_toh",
 "num_meses_pasa_cuo",
 "max_dias_atraso_deuda_directa",
 "max_dias_atraso_deuda_directa_consumo",
 "peor_calificacion_deuda_directa",
 "cnt_meses_pago_min_3m",
 "prom_mto_lineadisp_tc_consumo_toh_6um",
 "prom_Ratio_utilizacion_tc_disponibilidad_efectivo_9um",
 "prom_mto_lineadisp_tc_consumo_toh_9um",
 "prom_Ratio_utilizacion_tc_compras_ripley_3um",
 "prom_mto_lineadisp_tc_consumo_ripley_12um",
 "cant_entidades_deuda_directa_prestamo_personal_Consumo",
 "pct_pago_balance_3m",
 "flag_100normal_toh_9um",
 "prom_Ratio_utilizacion_tc_consumo_toh_6um",
 "frec_oec_12m",
 "max_max_dias_atraso_deuda_directa_consumo_falabella_12um",
 "maduracion",
 "Ratio_utilizacion_tc_consumo_ripley",
 "prom_mto_linea_tc_consumo_toh_9um",
 "prom_mto_deuda_directa_libre_disponibilidad_bcp_12um",
 "prom_Ratio_utilizacion_tc_consumo_ibk_3um",
 "frec_pro_12m",
 "Ratio_utilizacion_tc_compras_ripley",
 "flag_100normal_ripley_3um",
 "prom_mto_deuda_directa_prestamo_Norevolvente_bcp_12um",
 "mto_deuda_directa_tc_disponibilidad_efectivo_revolvente",
 "mto_lineadisp_tc_consumo_falabella",
 "pct_compra_balance_6m",
 "prom_mto_deuda_directa_toh_6um",
 "Ratio_utilizacion_tc_consumo_ibk",
 "prom_Ratio_utilizacion_tc_disponibilidad_efectivo_6um",
 "mto_deuda_directa_tc_disponibilidad_efectivo_toh"
]

# === Variables categóricas ===
VAR_CAT1 = [
    "flag_100normal_toh_12um",
    "flag_100normal_ripley_6um",
    "flag_100normal_toh_9um",
    "flag_100normal_ripley_3um"
]

VAR_CAT2 = [
    "peor_calificacion_deuda_directa"
]

# === Mapas de reemplazo ===
MAP_CAT1 = {
    "100%normal": 1,
    "no es 100%normal": 2
}

MAP_CAT2 = {
    "sin calificacion": 0,
    "normal": 1,
    "cpp": 2,
    "deficiente": 3,
    "dudoso": 4,
    "perdida": 5
}


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta el DataFrame de entrada a lo que necesita el modelo:
    - Crea columnas faltantes.
    - Convierte numéricas a float.
    - Transforma categóricas (mapas definidos).
    - Reordena columnas.
    """
    df_proc = df.copy()

    # Mapeo de categóricas VAR_CAT1
    for col in VAR_CAT1:
        try:
            df_proc[col] = df_proc[col].map(MAP_CAT1).fillna(0).astype(int)
        except Exception as e:
            print(f"[WARN] Error en mapeo VAR_CAT1 {col}: {e}")

    # Mapeo de categóricas VAR_CAT2
    for col in VAR_CAT2:
        try:
            df_proc[col] = df_proc[col].replace(MAP_CAT2).fillna(0).astype(int)
        except Exception as e:
            print(f"[WARN] Error en mapeo VAR_CAT2 {col}: {e}")

    for col in expected_features:
        if col not in df_proc:
            df_proc[col] = 0
    df_proc[expected_features] = df_proc[expected_features].fillna(0)

    # Conversión de numéricas a float
    for col in expected_features:
        try:
            df_proc[col] = df_proc[col].astype(float)
        except Exception as e:
            print(f"[WARN] No se pudo convertir {col} a float: {e}")        

    # Reordenar columnas
    df_proc = df_proc[expected_features]
    # Mapear nombres originales del CSV a los usados internamente

    return df_proc


def predict_probabilities(df: pd.DataFrame, model) -> pd.Series:
    """
    Realiza la predicción de probabilidades con el modelo cargado.
    :param df: DataFrame con los datos a predecir.
    :param model_path: Ruta al modelo .pkl
    :return: Serie con las probabilidades de la clase positiva.
    """
    df_prepared = preprocess_input(df)
    probabilities = model.predict_proba(df_prepared)[:, 1]
    return pd.Series(probabilities, name="prob_default")