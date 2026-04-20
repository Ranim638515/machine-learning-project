# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# ================================
# ✅ FIX : Chemins avec os.path.join (multi-OS)
# ================================
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_PATH = os.path.join(BASE_DIR, "models")
TRAIN_TEST_PATH = os.path.join(BASE_DIR, "data", "train_test")
CLEAN_PATH      = os.path.join(BASE_DIR, "data", "processed", "dataset_cleaned.csv")

print(f"[INFO] BASE_DIR    : {BASE_DIR}")
print(f"[INFO] MODELS_PATH : {MODELS_PATH}")

classifier     = None
regressor      = None
kmeans         = None
scaler_raw     = None
scaler_rfm     = None
CLF_COLS       = []
REG_COLS       = []
RFM_COLS       = []
NUM_COLS       = []
REG_USES_LOG   = False
MODELS_LOADED  = False

# ================================
# Chargement des modèles
# ================================
def load_models():
    global classifier, regressor, kmeans, scaler_raw, scaler_rfm
    global CLF_COLS, REG_COLS, RFM_COLS, NUM_COLS, REG_USES_LOG, MODELS_LOADED
    
    required = [
        "best_classifier.pkl",
        "best_regressor.pkl",
        "kmeans_model.pkl",
        "scaler_raw.pkl",
        "scaler_rfm.pkl",
        "clf_features.pkl",
        "reg_features.pkl",
        "rfm_features.pkl",
        "scaler_num_cols.pkl",
    ]
    
    # ✅ FIX : os.path.join correct
    for f in required:
        full_path = os.path.join(MODELS_PATH, f)
        if not os.path.exists(full_path):
            print(f"[AVERTISSEMENT] Fichier manquant : {full_path} — mode démo activé")
            return
    
    classifier   = joblib.load(os.path.join(MODELS_PATH, "best_classifier.pkl"))
    regressor    = joblib.load(os.path.join(MODELS_PATH, "best_regressor.pkl"))
    kmeans       = joblib.load(os.path.join(MODELS_PATH, "kmeans_model.pkl"))
    scaler_raw   = joblib.load(os.path.join(MODELS_PATH, "scaler_raw.pkl"))
    scaler_rfm   = joblib.load(os.path.join(MODELS_PATH, "scaler_rfm.pkl"))
    
    CLF_COLS.extend(joblib.load(os.path.join(MODELS_PATH, "clf_features.pkl")))
    REG_COLS.extend(joblib.load(os.path.join(MODELS_PATH, "reg_features.pkl")))
    RFM_COLS.extend(joblib.load(os.path.join(MODELS_PATH, "rfm_features.pkl")))
    NUM_COLS.extend(joblib.load(os.path.join(MODELS_PATH, "scaler_num_cols.pkl")))
    
    # Flag indiquant si le régresseur utilise log(y)
    log_flag_path = os.path.join(MODELS_PATH, "reg_uses_log.pkl")
    if os.path.exists(log_flag_path):
        REG_USES_LOG = joblib.load(log_flag_path)
    
    MODELS_LOADED = True
    print(f"[OK] Modèles chargés")
    print(f"     Classificateur : {type(classifier).__name__} ({len(CLF_COLS)} features)")
    print(f"     Régresseur     : {type(regressor).__name__} ({len(REG_COLS)} features, log={REG_USES_LOG})")
    print(f"     Clustering     : {type(kmeans).__name__} (k={kmeans.n_clusters})")
    print(f"     Features RFM   : {RFM_COLS}")

load_models()

# ================================
# Construction du DataFrame
# ================================
def build_full_dataframe(form_values: dict, col_list: list) -> pd.DataFrame:
    row = {col: 0.0 for col in col_list}
    for key, val in form_values.items():
        if key in row:
            row[key] = float(val)
    return pd.DataFrame([row])[col_list]

# ================================
# Niveaux de risque
# ================================
def risk_level(p):
    if p < 0.25:   return "Faible"
    elif p < 0.50: return "Moyen"
    elif p < 0.75: return "Élevé"
    else:          return "Critique"

# ================================
# Prédictions
# ================================
def predict_churn(form_values: dict):
    if not MODELS_LOADED:
        freq = float(form_values.get('Frequency', 1))
        prob = max(0.05, min(0.95, 0.5 - freq * 0.05 + np.random.uniform(-0.1, 0.1)))
        return {
            'churn_prediction':  1 if prob > 0.5 else 0,
            'churn_probability': round(prob * 100, 1),
            'churn_risk':        risk_level(prob),
        }
    
    # ✅ FIX : Appliquer le scaler UNIQUEMENT sur les colonnes numériques
    X_raw = build_full_dataframe(form_values, CLF_COLS)
    X_scaled = X_raw.copy()
    
    # Colonnes numériques à scaler (présentes dans CLF_COLS ∩ NUM_COLS)
    cols_to_scale = [c for c in NUM_COLS if c in CLF_COLS]
    
    if cols_to_scale:
        # Construire un DataFrame avec TOUTES les colonnes que le scaler attend
        # (il a été fit sur NUM_COLS complet)
        X_for_scaler = pd.DataFrame(0.0, index=[0], columns=NUM_COLS)
        for c in cols_to_scale:
            X_for_scaler[c] = X_raw[c].values
        
        X_scaled_full = pd.DataFrame(
            scaler_raw.transform(X_for_scaler),
            columns=NUM_COLS
        )
        # Récupérer uniquement les colonnes nécessaires au classificateur
        for c in cols_to_scale:
            X_scaled[c] = X_scaled_full[c].values
    
    X_scaled = X_scaled[CLF_COLS]
    pred = int(classifier.predict(X_scaled)[0])
    prob = float(classifier.predict_proba(X_scaled)[0, 1])
    return {
        'churn_prediction':  pred,
        'churn_probability': round(prob * 100, 1),
        'churn_risk':        risk_level(prob),
    }

def predict_monetary(form_values: dict):
    if not MODELS_LOADED:
        freq     = float(form_values.get('Frequency', 1))
        monetary = float(form_values.get('MonetaryTotal', 500))
        return round(abs(monetary * 10 + freq * 50), 2)
    
    # ✅ FIX : Appliquer le scaler comme pour le classificateur
    X_raw = build_full_dataframe(form_values, REG_COLS)
    X_scaled = X_raw.copy()
    
    cols_to_scale = [c for c in NUM_COLS if c in REG_COLS]
    
    if cols_to_scale:
        X_for_scaler = pd.DataFrame(0.0, index=[0], columns=NUM_COLS)
        for c in cols_to_scale:
            X_for_scaler[c] = X_raw[c].values
        
        X_scaled_full = pd.DataFrame(
            scaler_raw.transform(X_for_scaler),
            columns=NUM_COLS
        )
        for c in cols_to_scale:
            X_scaled[c] = X_scaled_full[c].values
    
    X_scaled = X_scaled[REG_COLS]
    
    # ✅ FIX : Si le régresseur utilise log(y), inverser la transformation
    pred_raw = float(regressor.predict(X_scaled)[0])
    if REG_USES_LOG:
        pred = np.expm1(pred_raw)
    else:
        pred = pred_raw
    
    pred = max(pred, 0)
    return round(pred, 2)

def predict_segment(form_values: dict):
    if not MODELS_LOADED:
        return np.random.choice(["Champions", "Réguliers", "Dormants", "VIP"])
    
    # ✅ FIX : Utiliser scaler_rfm dédié sur les 3 features RFM brutes
    rfm_values = {}
    for col in RFM_COLS:
        rfm_values[col] = float(form_values.get(col, 0))
    
    X_rfm = pd.DataFrame([rfm_values])[RFM_COLS]
    X_rfm_scaled = scaler_rfm.transform(X_rfm)
    
    cluster = int(kmeans.predict(X_rfm_scaled)[0])
    
    # Mapping auto basé sur les profils
    # (à ajuster selon les résultats réels de votre clustering)
    segment_names = {
        0: "Dormants",
        1: "Réguliers",
        2: "Champions",
        3: "VIP",
    }
    return segment_names.get(cluster, f"Cluster {cluster}")

# ================================
# Features dérivées
# ================================
def compute_derived_features(raw: dict) -> dict:
    freq      = float(raw.get('Frequency', 1))
    monetary  = float(raw.get('MonetaryTotal', 500))
    recency   = float(raw.get('Recency', 50))
    age       = float(raw.get('Age', 35))
    satisf    = float(raw.get('SatisfactionScore', 3))
    tickets   = float(raw.get('SupportTicketsCount', 1))
    qty       = float(raw.get('TotalQuantity', 10))
    ret_ratio = float(raw.get('ReturnRatio', 0.05))
    cancelled = float(raw.get('CancelledTransactions', 0))
    avg_days  = float(raw.get('AvgDaysBetweenPurchases', 30))
    
    monetary_avg = monetary / max(freq * 9, 1)
    monetary_std = monetary_avg * 1.02
    total_trans  = freq * 9
    unique_inv   = freq
    avg_lines    = total_trans / max(unique_inv, 1)
    unique_prod  = max(int(qty * 0.15), 1)
    
    return {
        'Frequency':                  freq,
        'MonetaryTotal':              monetary,
        'MonetaryAvg':                monetary_avg,
        'MonetaryStd':                monetary_std,
        'MonetaryMin':                monetary_avg * 0.33,
        'MonetaryMax':                monetary_avg * 3.5,
        'TotalQuantity':              qty,
        'AvgQuantityPerTransaction':  qty / max(total_trans, 1),
        'MinQuantity':                1.0,
        'MaxQuantity':                qty * 0.6,
        'FirstPurchaseDaysAgo':       avg_days * freq if avg_days > 0 else 362.0,
        'PreferredDayOfWeek':         2.0,
        'PreferredHour':              14.0,
        'WeekendPurchaseRatio':       0.28,
        'AvgDaysBetweenPurchases':    avg_days,
        'UniqueProducts':             float(unique_prod),
        'UniqueDescriptions':         float(unique_prod),
        'AvgProductsPerTransaction':  float(unique_prod),
        'UniqueCountries':            1.0,
        'NegativeQuantityCount':      0.0,
        'ZeroPriceCount':             0.0,
        'CancelledTransactions':      cancelled,
        'ReturnRatio':                ret_ratio,
        'TotalTransactions':          float(total_trans),
        'UniqueInvoices':             float(unique_inv),
        'AvgLinesPerInvoice':         avg_lines,
        'Age':                        age,
        'SupportTicketsCount':        tickets,
        'SatisfactionScore':          satisf,
        'Recency':                    recency,
        'AgeCategory':                2.0,
        'PreferredTimeOfDay':         2.0,
        'BasketSizeCategory':         1.0,
        'RegYear':                    2019.0,
        'RegMonth':                   6.0,
        'RegDay':                     15.0,
        'RegWeekday':                 2.0,
        'IP1':                        192.0,
        'IP2':                        168.0,
        'IP3':                        1.0,
        'IP4':                        10.0,
        'IP_Type_Public':             1.0,
        'FavoriteSeason_Hiver':       0.0,
        'FavoriteSeason_Été':         0.0,
        'Region_UK':                  1.0,
        'Region_Asie':                0.0,
        'Region_Autre':               0.0,
        'Region_Amérique du Nord':    0.0,
        'Region_Amérique du Sud':     0.0,
        'Region_Europe centrale':     0.0,
        'Region_Europe continentale': 0.0,
        "Region_Europe de l'Est":     0.0,
        'Region_Europe du Nord':      0.0,
        'Region_Europe du Sud':       0.0,
        'Region_Moyen-Orient':        0.0,
        'Region_Océanie':             0.0,
        'WeekendPreference_Semaine':  1.0,
        'WeekendPreference_Weekend':  0.0,
        'ProductDiversity_Modéré':    1.0,
        'ProductDiversity_Spécialisé':0.0,
        'Gender_M':                   0.0,
        'Gender_Unknown':             1.0,
        'AccountStatus_Closed':       0.0,
        'AccountStatus_Pending':      0.0,
        'AccountStatus_Suspended':    0.0,
        'Country_freq':               0.9,
    }

# ================================
# Routes Flask
# ================================
@app.route('/')
def index():
    return render_template('index.html', models_loaded=MODELS_LOADED)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data        = request.get_json()
        form_values = compute_derived_features(data)
        churn       = predict_churn(form_values)
        money       = predict_monetary(form_values)
        segment     = predict_segment(form_values)
        
        return jsonify({
            'success':            True,
            'churn_prediction':   churn['churn_prediction'],
            'churn_probability':  churn['churn_probability'],
            'churn_risk':         churn['churn_risk'],
            'monetary_predicted': money,
            'segment':            segment,
            'demo_mode':          not MODELS_LOADED,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/batch', methods=['POST'])
def batch_predict():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'Aucun fichier fourni'}), 400
        
        df      = pd.read_csv(request.files['file'])
        results = []
        
        for _, row in df.iterrows():
            fv      = compute_derived_features(row.to_dict())
            churn   = predict_churn(fv)
            money   = predict_monetary(fv)
            segment = predict_segment(fv)
            results.append({
                'churn_prediction':   churn['churn_prediction'],
                'churn_probability':  churn['churn_probability'],
                'churn_risk':         churn['churn_risk'],
                'monetary_predicted': money,
                'segment':            segment,
            })
        
        n            = len(results)
        churned      = sum(1 for r in results if r['churn_prediction'] == 1)
        avg_prob     = round(sum(r['churn_probability']  for r in results) / n, 1) if n else 0
        avg_monetary = round(sum(r['monetary_predicted'] for r in results) / n, 2) if n else 0
        segments_count = {}
        for r in results:
            s = r['segment']
            segments_count[s] = segments_count.get(s, 0) + 1
        
        return jsonify({
            'success':         True,
            'total':           n,
            'churned':         churned,
            'churn_rate':      round(churned / n * 100, 1) if n else 0,
            'avg_probability': avg_prob,
            'avg_monetary':    avg_monetary,
            'segments':        segments_count,
            'results':         results[:50],
            'demo_mode':       not MODELS_LOADED,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)