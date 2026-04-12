import pandas as pd
import numpy as np
import joblib
import os
import sys

# ================================
# 1️⃣ Chargement des modèles sauvegardés
# ================================
# Les modèles ont été entraînés et sauvegardés par train_model.py
# On les charge une seule fois au démarrage du script

MODELS_PATH = "models/"

# Vérifier que les modèles existent avant de charger
required_models = [
    "best_classifier.pkl",
    "best_regressor.pkl",
    "kmeans_model.pkl"
]

for model_file in required_models:
    path = MODELS_PATH + model_file
    if not os.path.exists(path):
        print(f"[ERREUR] Modèle introuvable : {path}")
        print("         Lancez d'abord : python src/train_model.py")
        sys.exit(1)

classifier = joblib.load(MODELS_PATH + "best_classifier.pkl")
regressor  = joblib.load(MODELS_PATH + "best_regressor.pkl")
kmeans     = joblib.load(MODELS_PATH + "kmeans_model.pkl")

print("[OK] Modèles chargés :")
print(f"     Classification : {type(classifier).__name__}")
print(f"     Régression     : {type(regressor).__name__}")
print(f"     Clustering     : {type(kmeans).__name__} (k={kmeans.n_clusters})")

# ================================
# 2️⃣ Chargement des données de référence
# ================================
# On charge X_train pour récupérer la liste exacte des colonnes
# et le dataset nettoyé pour la régression

X_train_ref = pd.read_csv("data/train_test/X_train.csv")
df_clean    = pd.read_csv("data/processed/dataset_cleaned.csv")

# Colonnes supprimées lors de la classification (leakage)
LEAKAGE_COLS = [
    'ChurnRiskCategory', 'LoyaltyLevel',
    'RFMSegment_Dormants', 'RFMSegment_Fidèles', 'RFMSegment_Potentiels',
    'CustomerType_Perdu', 'CustomerType_Nouveau',
    'Recency', 'CustomerTenureDays', 'SpendingCategory',
    'PreferredMonth', 'FavoriteSeason_Printemps',
]

# Colonnes attendues par le classificateur
clf_cols = [c for c in X_train_ref.columns if c not in LEAKAGE_COLS]

# Colonnes attendues par le régresseur
exclude_reg = ['MonetaryTotal', 'MonetaryAvg', 'MonetaryStd',
               'MonetaryMin', 'MonetaryMax', 'Churn', 'CustomerID', 'Country']
reg_cols = df_clean.drop(columns=[c for c in exclude_reg if c in df_clean.columns])\
                   .select_dtypes(include=[np.number]).columns.tolist()

print(f"\n[OK] Colonnes classificateur : {len(clf_cols)}")
print(f"     Colonnes régresseur     : {len(reg_cols)}")

# ================================
# 3️⃣ Fonctions de prédiction
# ================================

def predict_churn(X: pd.DataFrame) -> pd.DataFrame:
    """
    Prédit le risque de churn pour un ou plusieurs clients.

    Paramètres :
        X : DataFrame contenant les features des clients
            (après preprocessing, sans colonnes leakage)

    Retourne :
        DataFrame avec colonnes :
        - Churn_Prediction : 0 (fidèle) ou 1 (churné)
        - Churn_Probability : probabilité de churn (0.0 à 1.0)
        - Churn_Risk : niveau de risque texte (Faible/Moyen/Élevé/Critique)
    """
    # Garder uniquement les colonnes attendues par le modèle
    X_clf = X[[c for c in clf_cols if c in X.columns]].copy()

    # Remplir les colonnes manquantes par 0
    for col in clf_cols:
        if col not in X_clf.columns:
            X_clf[col] = 0

    X_clf = X_clf[clf_cols]

    # Prédiction de classe (0 ou 1)
    predictions  = classifier.predict(X_clf)

    # Probabilité de churn (colonne index 1 = classe "churné")
    probabilities = classifier.predict_proba(X_clf)[:, 1]

    # Niveau de risque selon la probabilité
    def risk_level(prob):
        if prob < 0.25:   return "Faible"
        elif prob < 0.50: return "Moyen"
        elif prob < 0.75: return "Élevé"
        else:             return "Critique"

    results = pd.DataFrame({
        'Churn_Prediction':  predictions,
        'Churn_Probability': probabilities.round(4),
        'Churn_Risk':        [risk_level(p) for p in probabilities],
    })

    return results


def predict_monetary(X: pd.DataFrame) -> pd.Series:
    """
    Prédit le montant total dépensé (MonetaryTotal) pour un ou plusieurs clients.

    Paramètres :
        X : DataFrame contenant les features des clients

    Retourne :
        Series avec les montants prédits en livres sterling (£)
    """
    X_reg = X[[c for c in reg_cols if c in X.columns]].copy()

    for col in reg_cols:
        if col not in X_reg.columns:
            X_reg[col] = 0

    X_reg = X_reg[reg_cols]

    predictions = regressor.predict(X_reg)
    predictions = np.maximum(predictions, 0)

    return pd.Series(predictions.round(2), name='MonetaryTotal_Predicted')


def predict_segment(X: pd.DataFrame) -> pd.Series:
    """
    Assigne un segment RFM à chaque client via K-Means.

    Paramètres :
        X : DataFrame contenant au minimum Frequency et MonetaryTotal
            (Recency optionnelle mais recommandée)

    Retourne :
        Series avec le numéro de cluster (0, 1, 2, 3) et le nom du segment
    """
    rfm_cols_available = [c for c in ['Recency', 'Frequency', 'MonetaryTotal']
                          if c in X.columns]

    if len(rfm_cols_available) < 2:
        print("[AVERTISSEMENT] Pas assez de colonnes RFM pour le clustering.")
        return pd.Series(['Inconnu'] * len(X), name='Segment')

    X_rfm = X[rfm_cols_available].copy()

    # Remplir les colonnes RFM manquantes par 0
    for col in ['Recency', 'Frequency', 'MonetaryTotal']:
        if col not in X_rfm.columns:
            X_rfm[col] = 0

    # Réordonner pour correspondre à l'ordre d'entraînement
    X_rfm = X_rfm[['Recency', 'Frequency', 'MonetaryTotal']]

    cluster_ids = kmeans.predict(X_rfm)

    # Noms des segments selon les profils identifiés dans train_model.py
    segment_names = {
        0: "Dormants",    # Recency élevée, faible activité
        1: "Réguliers",   # comportement moyen équilibré
        2: "Champions",   # très actifs, dépenses élevées
        3: "VIP",         # ultra-premium, dépenses extrêmes
    }

    segments = [segment_names.get(c, f"Cluster {c}") for c in cluster_ids]

    return pd.Series(segments, name='Segment')


# ================================
# 4️⃣ Prédiction sur le jeu de test
# ================================
# Démonstration : on prédit sur X_test pour vérifier la cohérence

print("\n" + "="*50)
print("DÉMONSTRATION — Prédictions sur X_test")
print("="*50)

X_test  = pd.read_csv("data/train_test/X_test.csv")
y_test  = pd.read_csv("data/train_test/y_test.csv").squeeze()

# --- Prédiction Churn ---
churn_results = predict_churn(X_test)
print(f"\n[Churn] Distribution des prédictions :")
print(churn_results['Churn_Prediction'].value_counts().to_string())
print(f"\n[Churn] Distribution des niveaux de risque :")
print(churn_results['Churn_Risk'].value_counts().to_string())

# Vérification cohérence avec y_test
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, churn_results['Churn_Prediction'])
print(f"\n[Churn] Accuracy sur X_test : {acc:.4f} ({acc*100:.1f}%)")

# --- Prédiction Monétaire ---
monetary_results = predict_monetary(X_test)


print(f"\n[Monétaire] Statistiques des montants prédits :")
print(f"  Minimum  : {monetary_results.min():.2f} £")
print(f"  Médiane  : {monetary_results.median():.2f} £")
print(f"  Moyenne  : {monetary_results.mean():.2f} £")
print(f"  Maximum  : {monetary_results.max():.2f} £")

# --- Segmentation ---
segment_results = predict_segment(X_test)
print(f"\n[Segment] Distribution des segments :")
print(segment_results.value_counts().to_string())

# ================================
# 5️⃣ Rapport final combiné
# ================================
# Assembler toutes les prédictions dans un seul DataFrame

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

# Sauvegarde du rapport complet
OUTPUT_PATH = "data/processed/predictions.csv"
rapport.to_csv(OUTPUT_PATH, index=False)
print(f"\n[OK] Rapport complet sauvegardé → {OUTPUT_PATH}")
print(f"     {len(rapport)} clients prédits")

# ================================
# 6️⃣ Exemple — Prédire UN nouveau client
# ================================
# Simulation d'un nouveau client qui n'était pas dans le dataset

print("\n" + "="*50)
print("EXEMPLE — Prédiction pour un nouveau client")
print("="*50)

# Créer un client fictif avec des valeurs typiques
nouveau_client = pd.DataFrame([{
    # Valeurs normalisées (StandardScaler — moyenne=0, std=1)
    # Client type "Régulier" : valeurs proches de 0
    'Frequency':                0.2,   # légèrement au-dessus de la moyenne
    'MonetaryTotal':            0.1,   # montant moyen
    'MonetaryAvg':             -0.1,
    'MonetaryStd':             -0.2,
    'MonetaryMin':              0.0,
    'MonetaryMax':              0.1,
    'TotalQuantity':            0.2,
    'AvgQuantityPerTransaction': 0.1,
    'MinQuantity':              0.0,
    'MaxQuantity':              0.1,
    'FirstPurchaseDaysAgo':    -0.3,   # client assez récent
    'PreferredDayOfWeek':       0.0,
    'PreferredHour':            0.2,
    'WeekendPurchaseRatio':    -0.1,
    'AvgDaysBetweenPurchases':  0.3,
    'UniqueProducts':           0.1,
    'UniqueDescriptions':       0.1,
    'AvgProductsPerTransaction': 0.0,
    'UniqueCountries':          0.0,
    'NegativeQuantityCount':   -0.2,
    'ZeroPriceCount':          -0.1,
    'CancelledTransactions':   -0.2,
    'ReturnRatio':             -0.1,
    'TotalTransactions':        0.2,
    'UniqueInvoices':           0.2,
    'AvgLinesPerInvoice':       0.0,
    'Age':                      0.1,
    'SupportTicketsCount':     -0.3,
    'SatisfactionScore':        0.4,
    'Recency':                 -0.4,   # client actif récemment
}])

churn_nc    = predict_churn(nouveau_client)
monetary_nc = predict_monetary(nouveau_client)
segment_nc  = predict_segment(nouveau_client)

print(f"\n  Churn prédit     : {'Churné' if churn_nc['Churn_Prediction'].iloc[0] == 1 else 'Fidèle'}")
print(f"  Probabilité      : {churn_nc['Churn_Probability'].iloc[0]*100:.1f}%")
print(f"  Niveau de risque : {churn_nc['Churn_Risk'].iloc[0]}")
print(f"  Montant prédit   : {monetary_nc.iloc[0]:.2f} £")
print(f"  Segment          : {segment_nc.iloc[0]}")

print("\n✅ predict.py terminé avec succès !")