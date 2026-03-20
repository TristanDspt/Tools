"""
Outils d'analyse de corrélations.

Choisir le bon test selon les types de variables :

+------------------+--------------------+------------------+----------------------+
| Type             | Test               | Indicateur       | Visualisation        |
+------------------+--------------------+------------------+----------------------+
| Quali / Quali    | Chi²               | χ², p-value      | Heatmap / Barplot    |
| Quanti / Quanti  | Pearson / Spearman | R², p-value      | Nuage de points      |
| Quanti / Quali   | ANOVA (ou KS)      | η², F, p-value   | Boxplot              |
+------------------+--------------------+------------------+----------------------+

Exemples :
    - Chi²    : Genre ↔ Catégories
    - Pearson : Âge ↔ Montant total
    - ANOVA   : Âge ↔ Catégories

Notes :
    - Pearson   : corrélation linéaire, sensible aux outliers.
    - Spearman  : version robuste basée sur les rangs, à préférer si distribution non normale.
    - KS (Kolmogorov-Smirnov) : alternative non-paramétrique à l'ANOVA.
    - p-value < 0.05 → corrélation statistiquement significative (seuil classique).
"""

# --------------------------------------------------------------------------------------------------------------------

# Import des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Bibliothèques statistiques
from scipy import stats
from scipy.stats import shapiro, normaltest, pearsonr, spearmanr

# Configuration
pd.set_option('display.max_columns', None)
plt.rcParams['figure.figsize'] = (10, 6)

def correlation_quanti_quanti(df, var1, var2, plot=True, test=True, report=True):

    # Step 1 : Vérifications
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{df} n'est pas un DataFrame")
    for col in [var1, var2]:
        if col not in df:
            raise TypeError(f"{col} ne fait pas partie du DataFrame")
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"{col} n'est pas un numeric : {df[col].dtype}")
    
    print("=" * 45)
    print("       ✅ VALIDATION OK")
    print("=" * 45)
    print(f"  DataFrame      : {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"  {var1:<15}: dtype={df[var1].dtype}, nulls={df[var1].isnull().sum()}")
    print(f"  {var2:<15}: dtype={df[var2].dtype}, nulls={df[var2].isnull().sum()}")
    print("=" * 45)

    # Step 2 : Test de normalité

    # Step 3 : TEST
    if test:
        corr_coef, p_value = stats.pearsonr(df[var1], df[var2])
    else:
        corr_coef, p_value = stats.spearmanr(df[var1], df[var2])
    r_carre = corr_coef ** 2
    
    # Step 4 : PLOT

    
    # Step 1 : Verifications
    """
        - df = DATAFRAME
        - col1 = column + dtype
        - col2 = idem
        if all OK print ok + résumé else KO
        """
    # Step 2 : Test normalité
    """
        - col1, col2 = NORMAL ?
        if OK : PEARSON else SPEARMAN
        fonction dédiée ? pour la v2
        """
    # Step 3 : TEST
    """
        - tests
        - p-value, R²
        """
    # Step 4 : PLOT
    """
        - scatter plot (x, y)
        """
    # Step 5 : REPORT