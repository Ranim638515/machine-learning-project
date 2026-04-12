import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
from xgboost import XGBClassifier, XGBRegressor

# ================================
# 1️⃣ Chargement des données
# ================================

TRAIN_TEST_PATH = "data/train_test/"
CLEAN_PATH      = "data/processed/dataset_cleaned.csv"
MODELS_PATH     = "models/"
REPORTS_PATH    = "reports/"

os.makedirs(MODELS_PATH,  exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

X_train = pd.read_csv(TRAIN_TEST_PATH + "X_train.csv")
X_test  = pd.read_csv(TRAIN_TEST_PATH + "X_test.csv")
y_train = pd.read_csv(TRAIN_TEST_PATH + "y_train.csv").squeeze()
y_test  = pd.read_csv(TRAIN_TEST_PATH + "y_test.csv").squeeze()

print(f"[OK] Données chargées")
print(f"     X_train : {X_train.shape} | X_test : {X_test.shape}")

# ================================
# 2️⃣ CLUSTERING K-MEANS
# ================================
# IMPORTANT : Le clustering se fait AVANT la suppression des colonnes leakage
# car Recency est une feature légitime pour segmenter les clients.
# On utilise les features RFM (Recency, Frequency, MonetaryTotal)

print("\n" + "="*50)
print("PARTIE 1 — CLUSTERING K-MEANS")
print("="*50)

rfm_cols = [c for c in ['Recency', 'Frequency', 'MonetaryTotal'] if c in X_train.columns]
X_rfm    = X_train[rfm_cols].copy()
print(f"     Features RFM utilisées : {rfm_cols}")

# Méthode Elbow : tester k=2 à k=9
inertias = []
k_range  = range(2, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_rfm)
    inertias.append(km.inertia_)
    print(f"  k={k} → inertie={km.inertia_:.0f}")

# Graphique Elbow
plt.figure(figsize=(8, 4))
plt.plot(list(k_range), inertias, 'o-', color='steelblue', linewidth=2, markersize=6)
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie")
plt.title("Méthode Elbow — choix du k optimal")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(REPORTS_PATH + "elbow_curve.png", dpi=150)
plt.close()
print(f"[OK] Graphique Elbow → {REPORTS_PATH}elbow_curve.png")

# Entraînement avec k=4
K_OPTIMAL = 4
kmeans = KMeans(n_clusters=K_OPTIMAL, random_state=42, n_init=10)
kmeans.fit(X_rfm)

sil_score = silhouette_score(X_rfm, kmeans.labels_)
print(f"\n[OK] K-Means k={K_OPTIMAL} | Silhouette={sil_score:.4f}")

cluster_counts = pd.Series(kmeans.labels_).value_counts().sort_index()
print(f"     Distribution : {cluster_counts.to_dict()}")

X_rfm_copy          = X_rfm.copy()
X_rfm_copy['Cluster'] = kmeans.labels_
print(f"\n     Profil moyen par cluster :")
print(X_rfm_copy.groupby('Cluster')[rfm_cols].mean().round(2).to_string())

# Visualisation clusters — axes selon features disponibles
x_axis = 'Recency'      if 'Recency'       in rfm_cols else rfm_cols[0]
y_axis = 'MonetaryTotal' if 'MonetaryTotal' in rfm_cols else rfm_cols[-1]

plt.figure(figsize=(8, 5))
colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']
for i in range(K_OPTIMAL):
    mask = kmeans.labels_ == i
    plt.scatter(
        X_rfm.loc[mask, x_axis],
        X_rfm.loc[mask, y_axis],
        c=colors[i], label=f'Cluster {i}', alpha=0.5, s=20
    )
plt.xlabel(f"{x_axis} (normalisée)")
plt.ylabel(f"{y_axis} (normalisée)")
plt.title(f"Segmentation K-Means — {x_axis} vs {y_axis}")
plt.legend()
plt.tight_layout()
plt.savefig(REPORTS_PATH + "kmeans_clusters.png", dpi=150)
plt.close()
print(f"[OK] Visualisation clusters → {REPORTS_PATH}kmeans_clusters.png")

joblib.dump(kmeans, MODELS_PATH + "kmeans_model.pkl")
print(f"[OK] Modèle K-Means sauvegardé → {MODELS_PATH}kmeans_model.pkl")

# ================================
# 3️⃣ SUPPRESSION COLONNES LEAKAGE
# ================================
# Ces colonnes sont supprimées UNIQUEMENT pour la classification/régression.
# Le clustering a déjà été fait avec les features complètes ci-dessus.
#
# Raison : ces colonnes ont été calculées à partir du Churn dans le dataset
# synthétique → les inclure donne 100% artificiel.

LEAKAGE_COLS = [
    # Calculées directement depuis Churn
    'ChurnRiskCategory',      # 0=Faible → 3=Critique
    'LoyaltyLevel',           # corrélé fortement au statut churn

    # RFMSegment One-Hot — définis par comportement churn
    'RFMSegment_Dormants',    # Dormant = churné
    'RFMSegment_Fidèles',     # Fidèle = non churné
    'RFMSegment_Potentiels',  # à risque de churn

    # CustomerType — "Perdu" = churné par définition
    'CustomerType_Perdu',
    'CustomerType_Nouveau',

    # Trop corrélées au Churn dans ce dataset synthétique (corr > 0.40)
    'Recency',                # corr=0.86
    'CustomerTenureDays',     # corr=0.45
    'SpendingCategory',       # corr=0.38
    'PreferredMonth',         # corr=0.43
    'FavoriteSeason_Printemps', # corr=0.30
]

leakage_present = [c for c in LEAKAGE_COLS if c in X_train.columns]
X_train_clf = X_train.drop(columns=leakage_present)
X_test_clf  = X_test.drop(columns=leakage_present)

print(f"\n[OK] {len(leakage_present)} colonnes leakage supprimées pour classification :")
for c in leakage_present:
    print(f"     - {c}")
print(f"     X_train_clf={X_train_clf.shape} | X_test_clf={X_test_clf.shape}")

# ================================
# 4️⃣ CLASSIFICATION — Prédire le Churn
# ================================

print("\n" + "="*50)
print("PARTIE 2 — CLASSIFICATION (Prédiction Churn)")
print("="*50)
print(f"     Features utilisées : {X_train_clf.shape[1]} colonnes")
print(f"     Distribution Churn train : {y_train.value_counts().to_dict()}")

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        C=0.1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2,
        random_state=42,
        eval_metric='logloss',
        verbosity=0
    ),
}

best_clf_name  = None
best_clf_f1    = 0
best_clf_model = None
results_clf    = {}

for name, clf in classifiers.items():
    clf.fit(X_train_clf, y_train)
    y_pred = clf.predict(X_test_clf)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')
    results_clf[name] = {'accuracy': acc, 'f1': f1}

    print(f"\n--- {name} ---")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1-Score : {f1:.4f}")
    print(classification_report(y_test, y_pred,
                                 target_names=['Fidèle (0)', 'Churné (1)']))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=['Fidèle', 'Churné']).plot(
        ax=ax, colorbar=False, cmap='Blues'
    )
    ax.set_title(f"Confusion — {name}")
    plt.tight_layout()
    safe_name = name.replace(' ', '_').lower()
    plt.savefig(REPORTS_PATH + f"confusion_{safe_name}.png", dpi=150)
    plt.close()

    if f1 > best_clf_f1:
        best_clf_f1    = f1
        best_clf_name  = name
        best_clf_model = clf

print(f"\n[OK] Meilleur classificateur : {best_clf_name} (F1={best_clf_f1:.4f})")

# Feature importance
if best_clf_name in ["Random Forest", "XGBoost"]:
    importances = best_clf_model.feature_importances_
    feat_df = pd.DataFrame({
        'feature':    X_train_clf.columns,
        'importance': importances
    }).sort_values('importance', ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=feat_df, y='feature', x='importance',
                hue='feature', palette='viridis', legend=False)
    plt.title(f"Top 15 features — {best_clf_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(REPORTS_PATH + "feature_importance_classification.png", dpi=150)
    plt.close()
    print(f"[OK] Feature importance → {REPORTS_PATH}feature_importance_classification.png")
    print(f"\n     Top 5 features pour prédire le Churn :")
    print(feat_df[['feature', 'importance']].head(5).to_string(index=False))

joblib.dump(best_clf_model, MODELS_PATH + "best_classifier.pkl")
print(f"[OK] Modèle sauvegardé → {MODELS_PATH}best_classifier.pkl")

# ================================
# 5️⃣ RÉGRESSION — Prédire MonetaryTotal
# ================================

print("\n" + "="*50)
print("PARTIE 3 — RÉGRESSION (Prédiction MonetaryTotal)")
print("="*50)

df_clean = pd.read_csv(CLEAN_PATH)

exclude = [
    'MonetaryTotal',  # target
    'MonetaryAvg',    # dérivé de MonetaryTotal
    'MonetaryStd',    # dérivé de MonetaryTotal
    'MonetaryMin',    # dérivé de MonetaryTotal
    'MonetaryMax',    # dérivé de MonetaryTotal
    'Churn',
    'CustomerID',
    'Country',
]
exclude_present = [c for c in exclude if c in df_clean.columns]

X_reg = df_clean.drop(columns=exclude_present)
y_reg = df_clean['MonetaryTotal']
X_reg = X_reg.select_dtypes(include=[np.number])

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
print(f"[OK] X_train={X_reg_train.shape} | X_test={X_reg_test.shape}")

regressors = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost Regressor": XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    ),
}

best_reg_name  = None
best_reg_r2    = -np.inf
best_reg_model = None

for name, reg in regressors.items():
    reg.fit(X_reg_train, y_reg_train)
    y_pred_reg = reg.predict(X_reg_test)

    mae  = mean_absolute_error(y_reg_test, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
    r2   = r2_score(y_reg_test, y_pred_reg)

    print(f"\n--- {name} ---")
    print(f"  MAE  : {mae:.2f} £")
    print(f"  RMSE : {rmse:.2f} £")
    print(f"  R²   : {r2:.4f}")

    if r2 > best_reg_r2:
        best_reg_r2    = r2
        best_reg_name  = name
        best_reg_model = reg

print(f"\n[OK] Meilleur régresseur : {best_reg_name} (R²={best_reg_r2:.4f})")

y_pred_best_reg = best_reg_model.predict(X_reg_test)
plt.figure(figsize=(7, 5))
plt.scatter(y_reg_test, y_pred_best_reg, alpha=0.3, s=15, color='steelblue')
max_val = float(max(y_reg_test.max(), y_pred_best_reg.max()))
plt.plot([0, max_val], [0, max_val], 'r--', linewidth=1.5, label='Prédiction parfaite')
plt.xlabel("Valeur réelle (£)")
plt.ylabel("Valeur prédite (£)")
plt.title(f"Régression — Réel vs Prédit ({best_reg_name})")
plt.legend()
plt.tight_layout()
plt.savefig(REPORTS_PATH + "regression_pred_vs_real.png", dpi=150)
plt.close()
print(f"[OK] Graphique → {REPORTS_PATH}regression_pred_vs_real.png")

joblib.dump(best_reg_model, MODELS_PATH + "best_regressor.pkl")
print(f"[OK] Modèle sauvegardé → {MODELS_PATH}best_regressor.pkl")

# ================================
# 6️⃣ RÉSUMÉ FINAL
# ================================

print("\n" + "="*50)
print("RÉSUMÉ FINAL")
print("="*50)
print(f"  Clustering    : K-Means k={K_OPTIMAL} | Silhouette={sil_score:.4f}")
for name, scores in results_clf.items():
    print(f"  {name:<25} | Accuracy={scores['accuracy']:.4f} | F1={scores['f1']:.4f}")
print(f"  Régression    : {best_reg_name} | R²={best_reg_r2:.4f}")
print(f"\n  Modèles → {MODELS_PATH}")
print(f"  Rapports → {REPORTS_PATH}")
print("\n✅ train_model.py terminé avec succès !")