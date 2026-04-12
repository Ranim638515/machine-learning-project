import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le rapport déjà généré par predict.py
rapport = pd.read_csv("data/processed/predictions.csv")

print(f"[OK] Rapport chargé : {rapport.shape[0]} clients")
print(rapport.head())

# --- Graphique 1 : Distribution des niveaux de risque ---
plt.figure(figsize=(7, 4))
risk_counts = rapport['Risque'].value_counts()
colors_risk = {'Faible': '#2e7d32', 'Moyen': '#f9a825',
               'Élevé': '#e65100', 'Critique': '#c62828'}
bars = plt.bar(risk_counts.index,
               risk_counts.values,
               color=[colors_risk.get(r, '#888888') for r in risk_counts.index],
               edgecolor='white', linewidth=0.8)
for bar, val in zip(bars, risk_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 5,
             str(val), ha='center', fontsize=11, fontweight='bold')
plt.title("Distribution des niveaux de risque Churn (875 clients)", fontsize=13)
plt.xlabel("Niveau de risque")
plt.ylabel("Nombre de clients")
plt.tight_layout()
plt.savefig("reports/predict_risk_distribution.png", dpi=150)
plt.close()
print("[OK] reports/predict_risk_distribution.png")

# --- Graphique 2 : Distribution des segments ---
plt.figure(figsize=(7, 4))
seg_counts = rapport['Segment'].value_counts()
colors_seg = {'Réguliers': '#1565c0', 'Dormants': '#c62828',
              'Champions': '#f9a825', 'VIP': '#6a1b9a'}
bars2 = plt.bar(seg_counts.index,
                seg_counts.values,
                color=[colors_seg.get(s, '#888888') for s in seg_counts.index],
                edgecolor='white', linewidth=0.8)
for bar, val in zip(bars2, seg_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 3,
             str(val), ha='center', fontsize=11, fontweight='bold')
plt.title("Distribution des segments clients (875 clients)", fontsize=13)
plt.xlabel("Segment")
plt.ylabel("Nombre de clients")
plt.tight_layout()
plt.savefig("reports/predict_segments.png", dpi=150)
plt.close()
print("[OK] reports/predict_segments.png")

# --- Graphique 3 : Distribution des probabilités ---
plt.figure(figsize=(8, 4))
plt.hist(rapport['Probabilité_Churn'], bins=30,
         color='steelblue', edgecolor='white', linewidth=0.5)
plt.axvline(x=0.25, color='#2e7d32', linestyle='--',
            linewidth=1.5, label='Seuil Faible (0.25)')
plt.axvline(x=0.50, color='#f9a825', linestyle='--',
            linewidth=1.5, label='Seuil Moyen (0.50)')
plt.axvline(x=0.75, color='#c62828', linestyle='--',
            linewidth=1.5, label='Seuil Critique (0.75)')
plt.title("Distribution des probabilités de churn", fontsize=13)
plt.xlabel("Probabilité de churn")
plt.ylabel("Nombre de clients")
plt.legend(fontsize=9)
plt.tight_layout()
plt.savefig("reports/predict_proba_distribution.png", dpi=150)
plt.close()
print("[OK] reports/predict_proba_distribution.png")

# --- Graphique 4 : Churn réel vs prédit ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
real_counts = rapport['Churn_Réel'].value_counts().sort_index()
axes[0].pie(real_counts.values,
            labels=['Fidèle (0)', 'Churné (1)'],
            colors=['#2e7d32', '#c62828'],
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 11})
axes[0].set_title("Churn Réel", fontsize=12, fontweight='bold')

pred_counts = rapport['Churn_Prédit'].value_counts().sort_index()
axes[1].pie(pred_counts.values,
            labels=['Fidèle (0)', 'Churné (1)'],
            colors=['#2e7d32', '#c62828'],
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 11})
axes[1].set_title("Churn Prédit", fontsize=12, fontweight='bold')
plt.suptitle("Comparaison Churn réel vs prédit", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("reports/predict_real_vs_pred.png", dpi=150)
plt.close()
print("[OK] reports/predict_real_vs_pred.png")

# --- Graphique 5 : Montants par segment ---
plt.figure(figsize=(8, 5))
rapport_nonzero = rapport[rapport['Montant_Prédit_£'] > 0]
sns.boxplot(data=rapport_nonzero,
            x='Segment', y='Montant_Prédit_£',
            order=['Dormants', 'Réguliers', 'Champions', 'VIP'],
            palette=['#c62828', '#1565c0', '#f9a825', '#6a1b9a'])
plt.title("Montants prédits par segment (clients > 0£)", fontsize=12)
plt.xlabel("Segment")
plt.ylabel("Montant prédit (£)")
plt.tight_layout()
plt.savefig("reports/predict_monetary_by_segment.png", dpi=150)
plt.close()
print("[OK] reports/predict_monetary_by_segment.png")

print("\n✅ Tous les graphiques générés dans reports/")