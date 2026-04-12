"""
utils.py — Fonctions utilitaires partagées
Utilisées par preprocessing.py, train_model.py et predict.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os


# ================================
# 1️⃣ FONCTIONS D'ANALYSE DES DONNÉES
# ================================

def analyse_dataset(df: pd.DataFrame) -> dict:
    """
    Génère un rapport complet sur la qualité du dataset.
    Retourne un dictionnaire avec toutes les statistiques.
    """
    rapport = {
        'nb_lignes':      df.shape[0],
        'nb_colonnes':    df.shape[1],
        'nb_doublons':    df.duplicated().sum(),
        'memoire_mb':     round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'types':          df.dtypes.value_counts().to_dict(),
        'manquants':      df.isnull().sum()[df.isnull().sum() > 0].to_dict(),
        'pct_manquants':  (df.isnull().mean() * 100)\
                          [df.isnull().mean() > 0].round(2).to_dict(),
    }

    print(f"[INFO] Dataset : {rapport['nb_lignes']} lignes × {rapport['nb_colonnes']} colonnes")
    print(f"       Mémoire : {rapport['memoire_mb']} MB")
    print(f"       Doublons : {rapport['nb_doublons']}")

    if rapport['manquants']:
        print(f"       Valeurs manquantes :")
        for col, nb in rapport['manquants'].items():
            pct = rapport['pct_manquants'][col]
            print(f"         - {col} : {nb} ({pct}%)")
    else:
        print(f"       Aucune valeur manquante.")

    return rapport


def detecter_outliers_iqr(df: pd.DataFrame, col: str) -> dict:
    """
    Détecte les valeurs aberrantes d'une colonne via la méthode IQR.
    Retourne les bornes et le nombre d'outliers.
    """
    Q1    = df[col].quantile(0.25)
    Q3    = df[col].quantile(0.75)
    IQR   = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    nb_outliers = ((df[col] < lower) | (df[col] > upper)).sum()

    return {
        'Q1':          Q1,
        'Q3':          Q3,
        'IQR':         IQR,
        'lower':       lower,
        'upper':       upper,
        'nb_outliers': nb_outliers,
        'pct_outliers': round(nb_outliers / len(df) * 100, 2)
    }


def corrélations_avec_target(df: pd.DataFrame,
                              target: str,
                              seuil: float = 0.3) -> pd.DataFrame:
    """
    Calcule les corrélations de toutes les features numériques
    avec la variable cible et signale les corrélations élevées.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target in num_cols:
        num_cols.remove(target)

    corr = df[num_cols].corrwith(df[target]).abs().sort_values(ascending=False)
    corr_df = pd.DataFrame({
        'feature':     corr.index,
        'correlation': corr.values.round(4)
    })

    leakage = corr_df[corr_df['correlation'] >= seuil]
    if not leakage.empty:
        print(f"[AVERTISSEMENT] {len(leakage)} features avec corrélation ≥ {seuil} avec '{target}' :")
        for _, row in leakage.iterrows():
            print(f"     - {row['feature']} : {row['correlation']}")

    return corr_df


# ================================
# 2️⃣ FONCTIONS DE VISUALISATION
# ================================

def plot_distribution(df: pd.DataFrame,
                      col: str,
                      output_path: str = None) -> None:
    """
    Affiche la distribution d'une colonne numérique
    avec histogramme + boxplot + statistiques.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogramme
    axes[0].hist(df[col].dropna(), bins=30,
                 color='steelblue', edgecolor='white', linewidth=0.5)
    axes[0].set_title(f"Distribution de {col}")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Fréquence")
    axes[0].axvline(df[col].mean(),   color='red',    linestyle='--',
                    linewidth=1.5, label=f"Moyenne={df[col].mean():.2f}")
    axes[0].axvline(df[col].median(), color='orange', linestyle='--',
                    linewidth=1.5, label=f"Médiane={df[col].median():.2f}")
    axes[0].legend(fontsize=9)

    # Boxplot
    axes[1].boxplot(df[col].dropna(), vert=False, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='steelblue'))
    axes[1].set_title(f"Boxplot de {col}")
    axes[1].set_xlabel(col)

    plt.suptitle(f"Analyse de la colonne : {col}", fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Graphique sauvegardé → {output_path}")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame,
                              output_path: str = None,
                              top_n: int = 20) -> None:
    """
    Génère une heatmap de corrélation pour les top_n colonnes
    les plus corrélées entre elles.
    """
    num_df = df.select_dtypes(include=[np.number])

    # Garder les top_n colonnes les plus corrélées
    corr_matrix = num_df.corr().abs()
    upper       = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    top_cols = upper.stack().nlargest(top_n).index
    cols_to_keep = list(set([c for pair in top_cols for c in pair]))[:top_n]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        num_df[cols_to_keep].corr(),
        annot=True, fmt='.2f',
        cmap='RdBu_r', center=0,
        linewidths=0.5, linecolor='white',
        annot_kws={'size': 8}
    )
    plt.title(f"Heatmap de corrélation — Top {top_n} features",
              fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Heatmap sauvegardée → {output_path}")
    plt.close()


def plot_churn_distribution(y: pd.Series,
                             output_path: str = None) -> None:
    """
    Visualise la distribution de la variable cible Churn.
    """
    counts = y.value_counts().sort_index()
    labels = ['Fidèle (0)', 'Churné (1)']
    colors = ['#2e7d32', '#c62828']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Barplot
    bars = axes[0].bar(labels, counts.values,
                       color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 10,
                     f'{val}\n({val/len(y)*100:.1f}%)',
                     ha='center', fontsize=11, fontweight='bold')
    axes[0].set_title("Distribution absolue")
    axes[0].set_ylabel("Nombre de clients")

    # Pie chart
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 11})
    axes[1].set_title("Distribution relative")

    plt.suptitle("Distribution de la variable cible Churn",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Graphique Churn → {output_path}")
    plt.close()


def plot_feature_importance(feature_names: list,
                             importances: np.ndarray,
                             model_name: str = "Modèle",
                             top_n: int = 15,
                             output_path: str = None) -> None:
    """
    Visualise les features les plus importantes d'un modèle arborescent.
    """
    feat_df = pd.DataFrame({
        'feature':    feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)

    plt.figure(figsize=(9, 6))
    sns.barplot(data=feat_df, y='feature', x='importance',
                hue='feature', palette='viridis', legend=False)
    plt.title(f"Top {top_n} features — {model_name}",
              fontsize=13, fontweight='bold')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Feature importance → {output_path}")
    plt.close()


# ================================
# 3️⃣ FONCTIONS D'ÉVALUATION
# ================================

def rapport_classification(y_true: pd.Series,
                            y_pred: np.ndarray,
                            model_name: str = "Modèle") -> dict:
    """
    Calcule et affiche toutes les métriques de classification.
    Retourne un dictionnaire avec les métriques principales.
    """
    from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                  recall_score, roc_auc_score)

    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted',
                                zero_division=0)
    recall    = recall_score(y_true, y_pred, average='weighted',
                             zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_pred)
    except Exception:
        auc = None

    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    if auc:
        print(f"  AUC-ROC   : {auc:.4f}")

    return {
        'model':     model_name,
        'accuracy':  round(acc, 4),
        'f1':        round(f1, 4),
        'precision': round(precision, 4),
        'recall':    round(recall, 4),
        'auc':       round(auc, 4) if auc else None,
    }


def rapport_regression(y_true: pd.Series,
                        y_pred: np.ndarray,
                        model_name: str = "Modèle") -> dict:
    """
    Calcule et affiche toutes les métriques de régression.
    Retourne un dictionnaire avec les métriques principales.
    """
    from sklearn.metrics import (mean_absolute_error,
                                  mean_squared_error, r2_score)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) /
                           np.where(y_true != 0, y_true, 1))) * 100

    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    print(f"  MAE  : {mae:.2f} £")
    print(f"  RMSE : {rmse:.2f} £")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f} %")

    return {
        'model': model_name,
        'mae':   round(mae, 2),
        'rmse':  round(rmse, 2),
        'r2':    round(r2, 4),
        'mape':  round(mape, 2),
    }


# ================================
# 4️⃣ FONCTIONS UTILITAIRES GÉNÉRALES
# ================================

def créer_dossiers(dossiers: list) -> None:
    """
    Crée les dossiers du projet s'ils n'existent pas.
    """
    for dossier in dossiers:
        os.makedirs(dossier, exist_ok=True)
        print(f"[OK] Dossier prêt : {dossier}")


def sauvegarder_résultats(résultats: list,
                           output_path: str) -> pd.DataFrame:
    """
    Sauvegarde une liste de dictionnaires de résultats en CSV.
    Utile pour comparer les performances de plusieurs modèles.
    """
    df_résultats = pd.DataFrame(résultats)
    df_résultats.to_csv(output_path, index=False)
    print(f"[OK] Résultats sauvegardés → {output_path}")
    print(df_résultats.to_string(index=False))
    return df_résultats


def résumé_dataset(df: pd.DataFrame, nom: str = "Dataset") -> None:
    """
    Affiche un résumé rapide et lisible du dataset.
    """
    print(f"\n{'='*50}")
    print(f"  RÉSUMÉ — {nom}")
    print(f"{'='*50}")
    print(f"  Dimensions    : {df.shape[0]} lignes × {df.shape[1]} colonnes")
    print(f"  Mémoire       : {df.memory_usage(deep=True).sum()/1024**2:.2f} MB")

    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'str']).columns
    print(f"  Numériques    : {len(num_cols)} colonnes")
    print(f"  Catégorielles : {len(cat_cols)} colonnes")

    manquants = df.isnull().sum().sum()
    print(f"  Valeurs NaN   : {manquants}")
    print(f"  Doublons      : {df.duplicated().sum()}")

    if 'Churn' in df.columns:
        taux = df['Churn'].mean() * 100
        print(f"  Taux Churn    : {taux:.1f}%")
    print(f"{'='*50}")


# ================================
# 5️⃣ TEST DES FONCTIONS (si lancé directement)
# ================================

if __name__ == "__main__":
    print("Test de utils.py...")

    # Charger les données
    df = pd.read_csv("data/processed/dataset_cleaned.csv")
    résumé_dataset(df, "dataset_cleaned.csv")

    # Test analyse
    rapport = analyse_dataset(df)

    # Test corrélations
    if 'Churn' in df.columns:
        corr_df = corrélations_avec_target(df, 'Churn', seuil=0.3)
        print(f"\nTop 5 corrélations avec Churn :")
        print(corr_df.head(5).to_string(index=False))

    # Test outliers
    if 'SupportTicketsCount' in df.columns:
        stats = detecter_outliers_iqr(df, 'SupportTicketsCount')
        print(f"\nOutliers SupportTicketsCount : {stats['nb_outliers']} ({stats['pct_outliers']}%)")

    # Test visualisation distribution Churn
    if 'Churn' in df.columns:
        plot_churn_distribution(df['Churn'],
                                output_path="reports/utils_churn_distribution.png")

    # Test heatmap corrélation
    plot_correlation_heatmap(df,
                             output_path="reports/utils_correlation_heatmap.png",
                             top_n=15)

    print("\n✅ utils.py — tous les tests passés !")