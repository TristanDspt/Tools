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
import plotly.express as px

# Bibliothèques statistiques
from scipy.stats import pearsonr, spearmanr

# Fichiers de dépendances
from data.utils_stats import verification_donnees, verification_normalite
from data.utils_viz import plot_normalite


def correlation_quanti_quanti(df, var1, var2, seuil=0.05, plot=True, report=True):
    """
    Analyse la corrélation entre deux variables quantitatives.
    Sélectionne automatiquement Pearson ou Spearman selon la normalité des distributions.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        var1 (str): Nom de la première variable quantitative.
        var2 (str): Nom de la deuxième variable quantitative.
        seuil (float): Seuil de significativité (par défaut 0.05).
        plot (bool): Affiche les graphiques si True (par défaut True).
        report (bool): Affiche le rapport final si True (par défaut True).

    Returns:
        None: Affiche les résultats dans la console.

    Example:
        >>> correlation_quanti_quanti(df, 'age', 'montant')
        >>> correlation_quanti_quanti(df, 'age', 'montant', seuil=0.01, plot=False)
    """
    
    # Initialisation de la fonction --------------------------------------
    print("=" * 70)
    print(f"📊 ANALYSE DE CORRÉLATION")
    print("=" * 70)


    # Step 1 : vérification préliminaire des données --------------------------------------
    print("\n" + "-" * 70)
    verif = verification_donnees(df, var1, var2)
    print(f"\n🔍 Analyse de la corrélation entre '{var1}' (type : {verif['dtype_var1']}) et '{var2}' (type : {verif['dtype_var2']})")
    
    if verif['type_df'] == 'DataFrame' and verif['dtype_var1'] in ['int64', 'float64'] and verif['dtype_var2'] in ['int64', 'float64'] and verif['shape'][0] > 2:
        print(f"✅ Vérification des données : OK, les variables sont quantitatives et le dataset est suffisant pour l'analyse.")
    else:
        print(f"❌ Vérification des données : Les variables doivent être quantitatives (int64 ou float64) et le dataset doit contenir plus de 2 lignes pour effectuer une analyse de corrélation.")
        return

    
    # Step 2 : Vérification de la normalité des variables --------------------------------------
    print("\n" + "-" * 70)
    print(f"\n🔍 Vérification de la normalité des variables '{var1}' et '{var2}'")
    normalite_var1 = verification_normalite(df, var1)
    normalite_var2 = verification_normalite(df, var2)
    
    # Affichage des résultats de normalité et graphique pour la variable 1
    print(f"P-value pour {normalite_var1['test']} '{var1}' : {normalite_var1['p_value']:.4f}")
    print(f"P-value pour {normalite_var2['test']} '{var2}' : {normalite_var2['p_value']:.4f}")
    if plot:
        plot_normalite(df, (var1, var2))
    
    if normalite_var1['p_value'] > seuil and normalite_var2['p_value'] > seuil:
        print(f"✅ Les deux variables suivent une distribution normale (p > {seuil}).")
        normalite = True
    else:
        print(f"⚠️ Au moins une des variables ne suit pas une distribution normale (p <= {seuil}).")
        normalite = False


    # Step 3 : Calcul de la corrélation --------------------------------------
    print("\n" + "-" * 70)

    if normalite:
        # Calcul de la corrélation de Pearson
        corr, p_value = pearsonr(df[var1], df[var2])
    else:
        # Calcul de la corrélation de Spearman
        corr, p_value = spearmanr(df[var1], df[var2])
    r_carre = corr ** 2
    
    if p_value < seuil:
        print(f"\n✅ La corrélation est statistiquement significative (p < {seuil}).")
    else:
        print(f"\n⚠️ La corrélation n'est pas statistiquement significative (p >= {seuil}).")


    # Step 4 : Visualisation de la corrélation --------------------------------------
    if plot: 
        # Nuage de points avec distributions marginales (jointplot)
        fig = px.scatter(
            df,
            x=var1,
            y=var2,
            trendline="ols",
            marginal_x='histogram',
            marginal_y='histogram',
            title=f'Distribution conjointe : {var1} et {var2}',
            labels={var1: f'{var1}', var2: f'{var2}'},
            opacity=0.5
        )

        fig.update_layout(template='plotly_white', height=600)
        fig.show()

    # Step 5 : Rapport & Conclusion --------------------------------------
    if report:
        print(f"=" * 70)
        print(f"📊 RAPPORT DE CORRÉLATION ENTRE '{var1}' ET '{var2}'")
        print("=" * 70)
        print(f"🔍 Type de corrélation utilisée : {'Pearson' if normalite else 'Spearman'}")
        print(f"📊 Coefficient de corrélation : {corr:.4f}, corrélation {'faible' if abs(corr) <= 0.3 else 'modérée' if 0.3 < abs(corr) <= 0.7 else 'forte'}")
        print(f"📊 P-value : {p_value:.4f}")
        print(f"📊 R² : {r_carre:.2f} -> '{var1}' explique {r_carre * 100:.0f}% de la variance de '{var2}'")
        if p_value < seuil:
            print(f"\n✅ CONCLUSION: Il existe une corrélation statistiquement significative entre '{var1}' et '{var2}' (p < {seuil}).")
        else:
            print(f"\n⚠️ CONCLUSION : Il n'existe pas de corrélation statistiquement significative entre '{var1}' et '{var2}' (p >= {seuil}).")