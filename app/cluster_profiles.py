import os, joblib, pathlib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from preprocessing import preprocess_input, expected_features, predict_probabilities  

DATA = "data/sample/base_var_comportamiento.csv"
OUT_DIR = "artifacts"
pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# Ruta a tu modelo entrenado supervisado (ajusta el path!)
MODEL_PATH = os.path.join(OUT_DIR, "comportamiento_xgb_VII_vf.pkl")

if __name__ == "__main__":
    df = pd.read_csv(DATA)

    # === Preprocesamiento igual que para el modelo ===
    X = preprocess_input(df)

    # === Cargar modelo supervisado ===
    model = joblib.load(MODEL_PATH) 

    # === Calcular probabilidad de default y añadir como feature extra ===
    prob_default = predict_probabilities(df, model)  
    X["prob_default"] = prob_default

    # === Clustering ===
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    kmeans.fit(Xs)

    joblib.dump(
        {"scaler": scaler, "kmeans": kmeans, "features": expected_features + ["prob_default"]},
        os.path.join(OUT_DIR, "cluster_model.pkl")
    )

    print("Saved clustering model ->", os.path.join(OUT_DIR, "cluster_model.pkl"))
