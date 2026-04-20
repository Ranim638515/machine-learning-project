## Conclusions EDA

- 4372 clients, 52 features, taux churn = 33.3%
- **Leakage détecté** : Recency (corr=0.859), ChurnRiskCategory, LoyaltyLevel
- Valeurs manquantes : Age (1311 NaN), AvgDaysBetweenPurchases (79 NaN)
- Valeurs sentinelles : SupportTicketsCount (-1, 999), SatisfactionScore (-1, 0, 99)
- Action : ces colonnes traitées dans preprocessing.py