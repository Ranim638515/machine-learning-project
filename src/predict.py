import pandas as pd
import numpy as np
import joblib
import os
import sys

# ================================
# 1️⃣ Chargement des modèles
# ================================
MODELS_PATH = "models/"

required_models = [
    "best_classifier.pkl",
    "best_regressor.pkl",
    "kmeans_model.pkl",
    "scaler_rfm.pkl",
    "clf_features.pkl",
    "reg_features.pkl",
    "rfm_features.pkl",
]

for model_file in required_models:
    path = MODELS_PATH + model_file
    if not os.path.exists(path):
        print(f"[ERREUR] Modèle introuvable : {path}")
        print("         Lancez d'abord : python src/train_model.py")
        sys.exit(1)

classifier  = joblib.load(MODELS_PATH + "best_classifier.pkl")
regressor   = joblib.load(MODELS_PATH + "best_regressor.pkl")
kmeans      = joblib.load(MODELS_PATH + "kmeans_model.pkl")
scaler_rfm  = joblib.load(MODELS_PATH + "scaler_rfm.pkl")
clf_cols    = joblib.load(MODELS_PATH + "clf_features.pkl")
reg_cols    = joblib.load(MODELS_PATH + "reg_features.pkl")
rfm_cols    = joblib.load(MODELS_PATH + "rfm_features.pkl")

# Flag log
log_flag_path = MODELS_PATH + "reg_uses_log.pkl"
REG_USES_LOG = joblib.load(log_flag_path) if os.path.exists(log_flag_path) else False

print("[OK] Modèles chargés :")
print(f"     Classification : {type(classifier).__name__}")
print(f"     Régression     : {type(regressor).__name__} (log={REG_USES_LOG})")
print(f"     Clustering     : {type(kmeans).__name__} (k={kmeans.n_clusters})")
print(f"\n[OK] Colonnes classificateur : {len(clf_cols)}")
print(f"     Colonnes régresseur     : {len(reg_cols)}")
print(f"     Colonnes RFM            : {rfm_cols}")

# ================================
# 2️⃣ Chargement des données
# ================================
X_test = pd.read_csv("data/train_test/X_test.csv")
y_test = pd.read_csv("data/train_test/y_test.csv").squeeze()
df_clean = pd.read_csv("data/processed/dataset_cleaned.csv")

# ================================
# 3️⃣ Fonctions de prédiction
# ================================
def predict_churn(X: pd.DataFrame) -> pd.DataFrame:
    X_clf = X[[c for c in clf_cols if c in X.columns]].copy()
    for col in clf_cols:
        if col not in X_clf.columns:
            X_clf[col] = 0
    X_clf = X_clf[clf_cols]
    
    predictions   = classifier.predict(X_clf)
    probabilities = classifier.predict_proba(X_clf)[:, 1]
    
    def risk_level(prob):
        if prob < 0.25:   return "Faible"
        elif prob < 0.50: return "Moyen"
        elif prob < 0.75: return "Élevé"
        else:             return "Critique"
    
    return pd.DataFrame({
        'Churn_Prediction':  predictions,
        'Churn_Probability': probabilities.round(4),
        'Churn_Risk':        [risk_level(p) for p in probabilities],
    })

def predict_monetary(X: pd.DataFrame) -> pd.Series:
    X_reg = X[[c for c in reg_cols if c in X.columns]].copy()
    for col in reg_cols:
        if col not in X_reg.columns:
            X_reg[col] = 0
    X_reg = X_reg[reg_cols]
    
    predictions_raw = regressor.predict(X_reg)
    
    # ✅ FIX : Inverser log si applicable
    if REG_USES_LOG:
        predictions = np.expm1(predictions_raw)
    else:
        predictions = predictions_raw
    
    predictions = np.maximum(predictions, 0)
    return pd.Series(predictions.round(2), name='MonetaryTotal_Predicted')

def predict_segment_from_raw(X_raw: pd.DataFrame) -> pd.Series:
    """
    ✅ FIX : Le clustering attend des valeurs RFM BRUTES (non normalisées)
    car scaler_rfm a été fit sur les données brutes.
    """
    available = [c for c in rfm_cols if c in X_raw.columns]
    if len(available) < len(rfm_cols):
        missing = set(rfm_cols) - set(available)
        print(f"[AVERTISSEMENT] Colonnes RFM manquantes : {missing}")
        X_rfm = X_raw[available].copy()
        for m in missing:
            X_rfm[m] = 0
    else:
        X_rfm = X_raw[rfm_cols].copy()
    
    X_rfm = X_rfm[rfm_cols]
    X_rfm_scaled = scaler_rfm.transform(X_rfm)
    cluster_ids = kmeans.predict(X_rfm_scaled)
    
    segment_names = {
        0: "Dormants",
        1: "Réguliers",
        2: "Champions",
        3: "VIP",
    }
    segments = [segment_names.get(c, f"Cluster {c}") for c in cluster_ids]
    return pd.Series(segments, name='Segment')

# ================================
# 4️⃣ Démonstration sur X_test
# ================================
print("\n" + "="*50)
print("DÉMONSTRATION — Prédictions sur X_test")
print("="*50)

# --- Churn ---
churn_results = predict_churn(X_test)
print(f"\n[Churn] Distribution des prédictions :")
print(churn_results['Churn_Prediction'].value_counts().to_string())
print(f"\n[Churn] Distribution des niveaux de risque :")
print(churn_results['Churn_Risk'].value_counts().to_string())

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, churn_results['Churn_Prediction'])
print(f"\n[Churn] Accuracy sur X_test : {acc:.4f} ({acc*100:.1f}%)")

# --- Monétaire ---
monetary_results = predict_monetary(X_test)
print(f"\n[Monétaire] Statistiques des montants prédits :")
print(f"  Minimum  : {monetary_results.min():.2f} £")
print(f"  Médiane  : {monetary_results.median():.2f} £")
print(f"  Moyenne  : {monetary_results.mean():.2f} £")
print(f"  Maximum  : {monetary_results.max():.2f} £")

# --- Segmentation ---
# ✅ FIX : Utiliser les données BRUTES (df_clean) pour le clustering
# On suit le même split (random_state=42 + stratify)
from sklearn.model_selection import train_test_split

cols_drop = ['Churn']
if 'CustomerID' in df_clean.columns:
    cols_drop.append('CustomerID')

X_full = df_clean.drop(columns=cols_drop)
y_full = df_clean['Churn']

_, X_test_raw, _, _ = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

segment_results = predict_segment_from_raw(X_test_raw.reset_index(drop=True))
print(f"\n[Segment] Distribution des segments :")
print(segment_results.value_counts().to_string())

# ================================
# 5️⃣ Rapport final
# ================================
print("\n" + "="*50)
print("RAPPORT FINAL — 10 premiers clients")
print("="*50)

rapport = pd.DataFrame({
    'Client_Index':        range(len(X_test)),
    'Churn_Réel':          y_test.values,
    'Churn_Prédit':        churn_results['Churn_Prediction'].values,
    'Probabilité_Churn':   churn_results['Churn_Probability'].values,
    'Risque':              churn_results['Churn_Risk'].values,
    'Montant_Prédit_£':    monetary_results.values,
    'Segment':             segment_results.values,
})

print(rapport.head(10).to_string(index=False))

OUTPUT_PATH = "data/processed/predictions.csv"
rapport.to_csv(OUTPUT_PATH, index=False)
print(f"\n[OK] Rapport complet sauvegardé → {OUTPUT_PATH}")
print(f"     {len(rapport)} clients prédits")

print("\n✅ predict.py terminé avec succès !")