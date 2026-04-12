"""
app.py — Application Flask pour la prédiction du Churn
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys

# ================================
# Chemins absolus — CORRECTION CLÉ
# ================================
# Peu importe d'où on lance le script, ces chemins fonctionnent toujours

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
MODELS_PATH   = os.path.join(BASE_DIR, "models/")
DATA_PATH     = os.path.join(BASE_DIR, "data/")

# Se placer à la racine du projet pour que tous les chemins relatifs fonctionnent
os.chdir(BASE_DIR)

print(f"[DEBUG] Racine projet    : {BASE_DIR}")
print(f"[DEBUG] Dossier templates : {TEMPLATES_DIR}")
print(f"[DEBUG] Dossier models   : {MODELS_PATH}")

# ================================
# Initialisation Flask
# ================================

app = Flask(__name__, template_folder=TEMPLATES_DIR)

# ================================
# Chargement des modèles
# ================================

classifier = joblib.load(os.path.join(MODELS_PATH, "best_classifier.pkl"))
regressor  = joblib.load(os.path.join(MODELS_PATH, "best_regressor.pkl"))
kmeans     = joblib.load(os.path.join(MODELS_PATH, "kmeans_model.pkl"))

# Colonnes attendues par le classificateur
X_train_ref = pd.read_csv(os.path.join(DATA_PATH, "train_test/X_train.csv"))

LEAKAGE_COLS = [
    'ChurnRiskCategory', 'LoyaltyLevel',
    'RFMSegment_Dormants', 'RFMSegment_Fidèles', 'RFMSegment_Potentiels',
    'CustomerType_Perdu', 'CustomerType_Nouveau',
    'Recency', 'CustomerTenureDays', 'SpendingCategory',
    'PreferredMonth', 'FavoriteSeason_Printemps',
]

CLF_COLS = [c for c in X_train_ref.columns if c not in LEAKAGE_COLS]

print(f"[OK] Modèles chargés : {type(classifier).__name__}")
print(f"[OK] {len(CLF_COLS)} features pour classification")

# ================================
# Fonction de prédiction
# ================================

def faire_prediction(data: dict) -> dict:
    df_client = pd.DataFrame([data])

    # ---- Churn ----
    X_clf = pd.DataFrame(0, index=[0], columns=CLF_COLS)
    for col in CLF_COLS:
        if col in df_client.columns:
            X_clf[col] = df_client[col].values[0]
    X_clf = X_clf.astype(float)

    churn_pred  = int(classifier.predict(X_clf)[0])
    churn_proba = float(classifier.predict_proba(X_clf)[0][1])

    if churn_proba < 0.25:
        risque, couleur = "Faible", "success"
    elif churn_proba < 0.50:
        risque, couleur = "Moyen", "warning"
    elif churn_proba < 0.75:
        risque, couleur = "Élevé", "orange"
    else:
        risque, couleur = "Critique", "danger"

    # ---- Montant ----
    df_clean = pd.read_csv(os.path.join(DATA_PATH, "processed/dataset_cleaned.csv"))
    exclude  = ['MonetaryTotal','MonetaryAvg','MonetaryStd',
                'MonetaryMin','MonetaryMax','Churn','CustomerID','Country']
    reg_cols = df_clean.drop(columns=[c for c in exclude if c in df_clean.columns])\
                       .select_dtypes(include=[np.number]).columns.tolist()

    X_reg = pd.DataFrame(0, index=[0], columns=reg_cols)
    for col in reg_cols:
        if col in df_client.columns:
            X_reg[col] = df_client[col].values[0]
    X_reg   = X_reg.astype(float)
    montant = max(0, float(regressor.predict(X_reg)[0]))

    # ---- Segment ----
    rfm_data   = pd.DataFrame([[
        float(data.get('Recency', 0)),
        float(data.get('Frequency', 0)),
        float(data.get('MonetaryTotal', 0))
    ]], columns=['Recency', 'Frequency', 'MonetaryTotal'])
    cluster_id = int(kmeans.predict(rfm_data)[0])
    segments   = {0: "Dormants", 1: "Réguliers", 2: "Champions", 3: "VIP"}
    segment    = segments.get(cluster_id, "Inconnu")

    return {
        'churn_prediction': churn_pred,
        'churn_label':      "Churné" if churn_pred == 1 else "Fidèle",
        'churn_proba':      round(churn_proba * 100, 1),
        'risque':           risque,
        'couleur':          couleur,
        'montant_predit':   round(montant, 2),
        'segment':          segment,
        'cluster_id':       cluster_id,
    }

# ================================
# Routes Flask
# ================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = {
                'Frequency':                 float(request.form.get('frequency', 0)),
                'MonetaryTotal':             float(request.form.get('monetary_total', 0)),
                'MonetaryAvg':               float(request.form.get('monetary_avg', 0)),
                'MonetaryStd':               float(request.form.get('monetary_std', 0)),
                'MonetaryMin':               float(request.form.get('monetary_min', 0)),
                'MonetaryMax':               float(request.form.get('monetary_max', 0)),
                'TotalQuantity':             float(request.form.get('total_quantity', 0)),
                'AvgQuantityPerTransaction': float(request.form.get('avg_qty', 0)),
                'MinQuantity':               float(request.form.get('min_qty', 0)),
                'MaxQuantity':               float(request.form.get('max_qty', 0)),
                'FirstPurchaseDaysAgo':      float(request.form.get('first_purchase', 0)),
                'PreferredDayOfWeek':        float(request.form.get('preferred_day', 0)),
                'PreferredHour':             float(request.form.get('preferred_hour', 0)),
                'WeekendPurchaseRatio':      float(request.form.get('weekend_ratio', 0)),
                'AvgDaysBetweenPurchases':   float(request.form.get('avg_days', 0)),
                'UniqueProducts':            float(request.form.get('unique_products', 0)),
                'UniqueDescriptions':        float(request.form.get('unique_desc', 0)),
                'AvgProductsPerTransaction': float(request.form.get('avg_products', 0)),
                'UniqueCountries':           float(request.form.get('unique_countries', 1)),
                'NegativeQuantityCount':     float(request.form.get('neg_qty', 0)),
                'ZeroPriceCount':            float(request.form.get('zero_price', 0)),
                'CancelledTransactions':     float(request.form.get('cancelled', 0)),
                'ReturnRatio':               float(request.form.get('return_ratio', 0)),
                'TotalTransactions':         float(request.form.get('total_transactions', 0)),
                'UniqueInvoices':            float(request.form.get('unique_invoices', 0)),
                'AvgLinesPerInvoice':        float(request.form.get('avg_lines', 0)),
                'Age':                       float(request.form.get('age', 49)),
                'SupportTicketsCount':       float(request.form.get('support_tickets', 0)),
                'SatisfactionScore':         float(request.form.get('satisfaction', 3)),
                'Recency':                   float(request.form.get('recency', 0)),
            }

        résultats = faire_prediction(data)

        if request.is_json:
            return jsonify({'status': 'success', 'predictions': résultats})
        else:
            return render_template('result.html', résultats=résultats, data=data)

    except Exception as e:
        if request.is_json:
            return jsonify({'status': 'error', 'message': str(e)}), 400
        else:
            return render_template('index.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'Corps JSON requis'}), 400
    résultats = faire_prediction(data)
    return jsonify({'status': 'success', 'predictions': résultats})


@app.route('/dashboard')
def dashboard():
    try:
        rapport = pd.read_csv(os.path.join(DATA_PATH, "processed/predictions.csv"))

        stats = {
            'total_clients': len(rapport),
            'nb_churnes':    int(rapport['Churn_Prédit'].sum()),
            'nb_fideles':    int((rapport['Churn_Prédit'] == 0).sum()),
            'taux_churn':    round(rapport['Churn_Prédit'].mean() * 100, 1),
            'montant_moyen': round(rapport['Montant_Prédit_£'].mean(), 2),
            'montant_total': round(rapport['Montant_Prédit_£'].sum(), 2),
            'segments':      rapport['Segment'].value_counts().to_dict(),
            'risques':       rapport['Risque'].value_counts().to_dict(),
        }
        return render_template('dashboard.html', stats=stats)

    except FileNotFoundError:
        return render_template('dashboard.html', stats=None,
                               error="Lancez d'abord : python src/predict.py")
    except Exception as e:
        return render_template('dashboard.html', stats=None,
                               error=f"Erreur : {str(e)}")


@app.route('/api/stats')
def api_stats():
    try:
        rapport = pd.read_csv(os.path.join(DATA_PATH, "processed/predictions.csv"))
        stats = {
            'total_clients': len(rapport),
            'nb_churnes':    int(rapport['Churn_Prédit'].sum()),
            'taux_churn':    round(rapport['Churn_Prédit'].mean() * 100, 1),
            'segments':      rapport['Segment'].value_counts().to_dict(),
            'risques':       rapport['Risque'].value_counts().to_dict(),
        }
        return jsonify({'status': 'success', 'stats': stats})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ================================
# Lancement
# ================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  Application Flask — Prédiction Churn")
    print("="*50)
    print(f"  URL       : http://localhost:5000")
    print(f"  Dashboard : http://localhost:5000/dashboard")
    print(f"  API       : http://localhost:5000/api/predict")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)