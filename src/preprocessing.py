import pandas as pd
import numpy as np
import sys
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# ================================
# 1️⃣ Chargement des données
# ================================

DATA_PATH = "data/raw/dataset ml.csv"
OUTPUT_PATH = "data/processed/dataset_cleaned.csv"

if not os.path.exists(DATA_PATH):
    print(f"[ERREUR] Fichier introuvable : {DATA_PATH}")
    sys.exit(1)

df = pd.read_csv(DATA_PATH)
print(f"[OK] Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(df.head())

# ================================
# 2️⃣ Analyse initiale des données
# ================================

print(df.info())
print(df.describe(include='all'))

missing = df.isnull().sum()
missing = missing[missing > 0]
print("Valeurs manquantes :")
print(missing if not missing.empty else "Aucune valeur manquante détectée")

nb_duplicates = df.duplicated().sum()
print(f"Nombre de doublons : {nb_duplicates}")

# ================================
# 3️⃣ Nettoyage des données
# ================================

df = df.drop_duplicates()
print(f"[OK] Doublons supprimés. Lignes restantes : {len(df)}")

# Imputation Age par la médiane (30% de NaN, robuste aux valeurs extrêmes)
if 'Age' in df.columns:
    nb_missing_age = df['Age'].isnull().sum()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    print(f"[OK] 'Age' : {nb_missing_age} NaN imputés par la médiane ({df['Age'].median():.1f})")
else:
    print("[AVERTISSEMENT] Colonne 'Age' introuvable.")

# Imputation AvgDaysBetweenPurchases par la médiane (79 NaN détectés)
if 'AvgDaysBetweenPurchases' in df.columns:
    nb_missing = df['AvgDaysBetweenPurchases'].isnull().sum()
    df['AvgDaysBetweenPurchases'] = df['AvgDaysBetweenPurchases'].fillna(df['AvgDaysBetweenPurchases'].median())
    print(f"[OK] 'AvgDaysBetweenPurchases' : {nb_missing} NaN imputés par la médiane")

# ================================
# 4️⃣ Valeurs sentinelles + aberrantes
# ================================

# --- SupportTicketsCount ---
COL_TICKETS = 'SupportTicketsCount'
if COL_TICKETS in df.columns:
    nb_sentinel = df[COL_TICKETS].isin([-1, 999]).sum()
    df[COL_TICKETS] = df[COL_TICKETS].replace([-1, 999], np.nan)
    print(f"[OK] '{COL_TICKETS}' : {nb_sentinel} valeurs sentinelles (-1, 999) → NaN")

    nb_nan = df[COL_TICKETS].isnull().sum()
    df[COL_TICKETS] = df[COL_TICKETS].fillna(df[COL_TICKETS].median())
    print(f"[OK] '{COL_TICKETS}' : {nb_nan} NaN imputés par la médiane ({df[COL_TICKETS].median():.1f})")

    Q1, Q3 = df[COL_TICKETS].quantile(0.25), df[COL_TICKETS].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    nb_outliers = ((df[COL_TICKETS] < lower) | (df[COL_TICKETS] > upper)).sum()
    df[COL_TICKETS] = df[COL_TICKETS].clip(lower, upper)
    print(f"[OK] '{COL_TICKETS}' : {nb_outliers} outliers → capping [{lower:.2f}, {upper:.2f}]")
else:
    print(f"[AVERTISSEMENT] Colonne '{COL_TICKETS}' introuvable.")

# --- SatisfactionScore ---
COL_SATIS = 'SatisfactionScore'
if COL_SATIS in df.columns:
    nb_sentinel_s = df[COL_SATIS].isin([-1, 0, 99]).sum()
    df[COL_SATIS] = df[COL_SATIS].replace([-1, 0, 99], np.nan)
    print(f"[OK] '{COL_SATIS}' : {nb_sentinel_s} valeurs sentinelles (-1, 0, 99) → NaN")

    nb_nan_s = df[COL_SATIS].isnull().sum()
    df[COL_SATIS] = df[COL_SATIS].fillna(df[COL_SATIS].median())
    print(f"[OK] '{COL_SATIS}' : {nb_nan_s} NaN imputés par la médiane ({df[COL_SATIS].median():.1f})")
else:
    print(f"[AVERTISSEMENT] Colonne '{COL_SATIS}' introuvable.")

# ================================
# 5️⃣ Feature Engineering — RegistrationDate
# ================================

if 'RegistrationDate' in df.columns:
    nb_before = df['RegistrationDate'].isnull().sum()
    df["RegistrationDate"] = pd.to_datetime(df["RegistrationDate"], format="mixed", dayfirst=True, errors='coerce')
    nb_after = df['RegistrationDate'].isnull().sum()
    nb_failed = nb_after - nb_before

    df['RegYear']    = df['RegistrationDate'].dt.year
    df['RegMonth']   = df['RegistrationDate'].dt.month
    df['RegDay']     = df['RegistrationDate'].dt.day
    df['RegWeekday'] = df['RegistrationDate'].dt.weekday

    df = df.drop(columns=["RegistrationDate"])
    print(f"[OK] 'RegistrationDate' : features extraites + colonne supprimée ({nb_failed} NaT)")
else:
    print("[AVERTISSEMENT] Colonne 'RegistrationDate' introuvable.")

# ================================
# 6️⃣ Suppression colonnes inutiles
# ================================

if 'NewsletterSubscribed' in df.columns:
    df = df.drop(columns=["NewsletterSubscribed"])
    print("[OK] 'NewsletterSubscribed' supprimée (variance nulle).")

# ================================
# 7️⃣ Feature Engineering — LastLoginIP
# ================================

if 'LastLoginIP' in df.columns:
    df[["IP1","IP2","IP3","IP4"]] = df["LastLoginIP"].str.split(".", expand=True).astype(int)

    def ip_type(ip):
        first = int(ip.split('.')[0])
        return "Private" if first in [10, 172, 192] else "Public"

    df["IP_Type"] = df["LastLoginIP"].apply(ip_type)
    df = df.drop(columns=["LastLoginIP"])
    print(f"[OK] IP features créées : IP1-4 + IP_Type\n{df['IP_Type'].value_counts()}")
else:
    print("[AVERTISSEMENT] Colonne 'LastLoginIP' introuvable.")

# ================================
# 8️⃣ Analyse colonnes catégorielles
# ================================

cat_cols = df.select_dtypes(include=['str', 'object']).columns.tolist()
print(f"\n[INFO] Colonnes catégorielles ({len(cat_cols)}) : {cat_cols}")

for col in cat_cols:
    print(f"\n--- Distribution de '{col}' ---")
    print(df[col].value_counts())
    print(f"  Valeurs manquantes : {df[col].isnull().sum()}")

# ================================
# 9️⃣ Encodage des variables catégorielles
# ================================

# --- Encodage Ordinal ---
# IMPORTANT : LoyaltyLevel et ChurnRiskCategory sont encodés ici
# mais seront supprimés juste après (leakage)
# On les encode quand même pour ne pas casser le pipeline
ordinal_mappings = {
    'AgeCategory':        ['Inconnu', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
    'SpendingCategory':   ['Low', 'Medium', 'High', 'VIP'],
    'PreferredTimeOfDay': ['Nuit', 'Matin', 'Midi', 'Après-midi', 'Soir'],
    'LoyaltyLevel':       ['Nouveau', 'Jeune', 'Établi', 'Ancien'],
    'ChurnRiskCategory':  ['Faible', 'Moyen', 'Élevé', 'Critique'],
    'BasketSizeCategory': ['Petit', 'Moyen', 'Grand'],
}

for col, categories in ordinal_mappings.items():
    if col in df.columns:
        df[col] = df[col].fillna(categories[0])
        enc = OrdinalEncoder(
            categories=[categories],
            handle_unknown='use_encoded_value',
            unknown_value=-1
        )
        df[[col]] = enc.fit_transform(df[[col]])
        print(f"[OK] Encodage ordinal → '{col}'")
    else:
        print(f"[AVERTISSEMENT] Colonne '{col}' introuvable.")

# --- Encodage One-Hot ---
onehot_cols = [
    'CustomerType',
    'FavoriteSeason',
    'Region',
    'WeekendPreference',
    'ProductDiversity',
    'Gender',
    'AccountStatus',
    'IP_Type',
]

onehot_present = [c for c in onehot_cols if c in df.columns]
if onehot_present:
    df = pd.get_dummies(df, columns=onehot_present, drop_first=True)
    print(f"[OK] One-Hot Encoding → {onehot_present}")
    print(f"     Dimensions après One-Hot : {df.shape}")

# RFMSegment : One-Hot
# IMPORTANT : sera supprimé juste après (leakage) — on encode d'abord
if 'RFMSegment' in df.columns:
    df = pd.get_dummies(df, columns=['RFMSegment'], drop_first=True)
    print("[OK] One-Hot Encoding → 'RFMSegment'")

# Country : encodage par fréquence
if 'Country' in df.columns:
    freq_map = df['Country'].value_counts(normalize=True)
    df['Country_freq'] = df['Country'].map(freq_map)
    df = df.drop(columns=['Country'])
    print("[OK] 'Country' → encodage par fréquence (Country_freq)")

# ================================
# 🔟 Suppression colonnes LEAKAGE
# ================================
# Ces colonnes contiennent déjà la réponse (Churn) → le modèle "triche"
# Recency        : client parti = plus d'achat = Recency élevée (corr=0.859)
# ChurnRiskCategory : calculée directement depuis Churn
# LoyaltyLevel   : lié au statut churn
# RFMSegment_*   : Dormant = churné par définition

leakage_cols = [
    'Recency',
    'ChurnRiskCategory',
    'LoyaltyLevel',
    'RFMSegment_Dormants',
    'RFMSegment_Fidèles',
    'RFMSegment_Potentiels',
]

leakage_present = [c for c in leakage_cols if c in df.columns]
df = df.drop(columns=leakage_present)
print(f"\n[OK] {len(leakage_present)} colonnes leakage supprimées : {leakage_present}")
print(f"     Dimensions après suppression leakage : {df.shape}")

# ================================
# 1️⃣1️⃣ Sauvegarde dataset nettoyé
# ================================

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\n[OK] dataset_cleaned.csv → {OUTPUT_PATH}")
print(f"     Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")

# ================================
# 1️⃣2️⃣ Séparation X / y
# ================================

if 'Churn' not in df.columns:
    print("[ERREUR] Colonne cible 'Churn' introuvable.")
    sys.exit(1)

cols_to_drop = ['Churn']
if 'CustomerID' in df.columns:
    cols_to_drop.append('CustomerID')

X = df.drop(columns=cols_to_drop)
y = df['Churn']

print(f"\n[INFO] X : {X.shape} | y : {y.shape}")
print(f"[INFO] Distribution Churn :\n{y.value_counts()}")
print(f"[INFO] Taux de churn : {y.mean()*100:.1f}%")

# ================================
# 1️⃣3️⃣ Train/Test Split (80/20 stratifié)
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"[OK] Split : Train {X_train.shape} | Test {X_test.shape}")

# ================================
# 1️⃣4️⃣ Normalisation — StandardScaler
# ================================

# RÈGLE ABSOLUE :
#   fit_transform sur X_train uniquement
#   transform seulement sur X_test
#   ne jamais normaliser y

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols]  = scaler.transform(X_test[num_cols])

print(f"[OK] StandardScaler sur {len(num_cols)} colonnes numériques")
print(f"     Moyenne X_train après scaling (doit être ~0) : {X_train[num_cols].mean().mean():.4f}")

MODELS_PATH = "models/"
os.makedirs(MODELS_PATH, exist_ok=True)

joblib.dump(scaler, MODELS_PATH + "scaler_raw.pkl")
print(f"[OK] Scaler sauvegardé → {MODELS_PATH}scaler_raw.pkl")

joblib.dump(num_cols, MODELS_PATH + "scaler_num_cols.pkl")
print(f"[OK] Liste colonnes numériques sauvegardée → {MODELS_PATH}scaler_num_cols.pkl")

# ================================
# 1️⃣5️⃣ ACP — Analyse en Composantes Principales
# ================================
# Objectif : analyser combien de composantes expliquent ≥ 95% de la variance
# RÈGLE : fit sur X_train uniquement, transform sur X_test

os.makedirs("reports/", exist_ok=True)

# Fit PCA complète pour analyser la variance (sur colonnes numériques uniquement)
pca_full = PCA(random_state=42)
pca_full.fit(X_train[num_cols])

# Variance cumulée
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = int(np.argmax(cumvar >= 0.95)) + 1
n_components_99 = int(np.argmax(cumvar >= 0.99)) + 1

print(f"\n[ACP] Composantes pour expliquer 95% de la variance : {n_components_95}")
print(f"[ACP] Composantes pour expliquer 99% de la variance : {n_components_99}")
print(f"[ACP] Variance expliquée par les 10 premières composantes : {cumvar[min(9, len(cumvar)-1)]:.4f}")

# Graphique variance cumulée
plt.figure(figsize=(9, 5))
plt.plot(range(1, len(cumvar) + 1), cumvar, 'o-',
         color='steelblue', linewidth=1.5, markersize=3)
plt.axhline(y=0.95, color='red',    linestyle='--', linewidth=1.5, label='Seuil 95%')
plt.axhline(y=0.99, color='orange', linestyle='--', linewidth=1.5, label='Seuil 99%')
plt.axvline(x=n_components_95, color='red',    linestyle=':', linewidth=1, alpha=0.7)
plt.axvline(x=n_components_99, color='orange', linestyle=':', linewidth=1, alpha=0.7)
plt.xlabel("Nombre de composantes")
plt.ylabel("Variance cumulée expliquée")
plt.title("ACP — Variance expliquée cumulée")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("reports/pca_variance.png", dpi=150)
plt.close()
print(f"[OK] Graphique ACP → reports/pca_variance.png")

# Application de l'ACP avec seuil 95% de variance
# IMPORTANT : on conserve X_train/X_test originaux pour les modèles basés sur les arbres
# L'ACP est sauvegardée séparément pour usage optionnel
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train[num_cols])
X_test_pca  = pca.transform(X_test[num_cols])

print(f"[OK] ACP appliquée : {len(num_cols)} features → {pca.n_components_} composantes")
print(f"     Variance expliquée : {pca.explained_variance_ratio_.sum():.4f}")

# Sauvegarde du modèle PCA
joblib.dump(pca, MODELS_PATH + "pca_model.pkl")
print(f"[OK] Modèle PCA sauvegardé → {MODELS_PATH}pca_model.pkl")

# ================================
# 1️⃣6️⃣ Sauvegarde Train/Test
# ================================

TRAIN_TEST_PATH = "data/train_test/"
os.makedirs(TRAIN_TEST_PATH, exist_ok=True)

X_train.to_csv(TRAIN_TEST_PATH + "X_train.csv", index=False)
X_test.to_csv(TRAIN_TEST_PATH  + "X_test.csv",  index=False)
y_train.to_csv(TRAIN_TEST_PATH + "y_train.csv", index=False)
y_test.to_csv(TRAIN_TEST_PATH  + "y_test.csv",  index=False)

# Sauvegarde des versions PCA (optionnel — pour modèles linéaires ou SVM)
np.save(TRAIN_TEST_PATH + "X_train_pca.npy", X_train_pca)
np.save(TRAIN_TEST_PATH + "X_test_pca.npy",  X_test_pca)

print(f"\n[OK] Fichiers sauvegardés dans '{TRAIN_TEST_PATH}'")
print(f"     X_train : {X_train.shape} | X_test : {X_test.shape}")
print(f"     X_train_pca : {X_train_pca.shape} | X_test_pca : {X_test_pca.shape}")
print(f"     y_train : {y_train.shape} | y_test : {y_test.shape}")
print("\n✅ Pipeline preprocessing terminé avec succès !")