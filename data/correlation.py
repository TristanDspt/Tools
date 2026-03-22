
# Import des bibliothèques
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Bibliothèques statistiques
from scipy.stats import pearsonr, spearmanr, chi2_contingency, ttest_ind, mannwhitneyu, f_oneway, levene, kruskal

# Fichiers de dépendances
from data.utils_stats import verification_donnees, verification_normalite, cohen_d, eta_squared
from data.utils_viz import plot_normalite


def corr_quanti_quanti(df, var1, var2, seuil=0.05, plot=True, report=True):
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
    
    liste_quanti = ['int64', 'float64']
    if verif['type_df'] == 'DataFrame' and verif['dtype_var1'] in liste_quanti and verif['dtype_var2'] in liste_quanti and verif['shape'][0] > 2:
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



def corr_quali_quali(df, var1, var2, seuil=0.05, plot=True, report=True):
    """
    Analyse la corrélation entre deux variables qualitatives via le test du Chi².
    Calcule le V de Cramer pour mesurer l'intensité du lien.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        var1 (str): Nom de la première variable qualitative.
        var2 (str): Nom de la deuxième variable qualitative.
        seuil (float): Seuil de significativité (par défaut 0.05).
        plot (bool): Affiche les graphiques si True (par défaut True).
        report (bool): Affiche le rapport final si True (par défaut True).

    Returns:
        None: Affiche les résultats dans la console.

    Example:
        >>> correlation_quali_quali(df, 'genre', 'categorie')
        >>> correlation_quali_quali(df, 'genre', 'categorie', seuil=0.01, plot=False)
    """
    # Initialisation de la fonction --------------------------------------
    print("=" * 70)
    print(f"📊 ANALYSE DE CORRÉLATION")
    print("=" * 70)


    # Step 1 : vérification préliminaire des données --------------------------------------
    print("\n" + "-" * 70)
    verif = verification_donnees(df, var1, var2)
    print(f"\n🔍 Analyse de la corrélation entre '{var1}' (type : {verif['dtype_var1']}) et '{var2}' (type : {verif['dtype_var2']})")
    
    liste_quali = ['object', 'category']
    if verif['type_df'] == 'DataFrame' and verif['dtype_var1'] in liste_quali and verif['dtype_var2'] in liste_quali and verif['shape'][0] > 2:
        print(f"✅ Vérification des données : OK, les variables sont qualitatives et le dataset est suffisant pour l'analyse.")
    else:
        print(f"❌ Vérification des données : Les variables doivent être qualitatives (object ou category) et le dataset doit contenir plus de 2 lignes pour effectuer une analyse de corrélation.")
        return    


    # Step 2 : Vérification de la normalité des variables --------------------------------------
    # None


    # Step 3 : Calcul de la corrélation --------------------------------------
    print("\n" + "-" * 70)

    tab = pd.crosstab(df[var1], df[var2])
    chi2, p_value, dof, exp = chi2_contingency(tab)
    v_cramer = np.sqrt(chi2 / (df.shape[0] * (min(df[var1].nunique(), df[var2].nunique()) - 1)))
    
    if p_value < seuil:
        print(f"\n✅ La corrélation est statistiquement significative (p < {seuil}).")
    else:
        print(f"\n⚠️ La corrélation n'est pas statistiquement significative (p >= {seuil}).")


    # Step 4 : Visualisation de la corrélation --------------------------------------
    if plot:
        if max(df[var1].nunique(), df[var2].nunique()) <= 5:
            # Barplot
            tab_graph = tab.reset_index().melt(id_vars=var1).copy()

            sns.barplot(
                tab_graph,
                x=var2,
                y='value',
                hue=var1,
                palette='viridis'
            )
        else:
            # Heatmap
            sns.heatmap(
                tab,
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=tab.columns,
                yticklabels=tab.index, 
                linewidths=2
                )

        plt.title("Corrélation Chi2", fontname='Arial' , fontweight='bold', fontsize=16, x=0.51, ha='center')
        plt.tick_params(axis='x', labelsize=11.5)
        plt.tick_params(axis='y', labelsize=11.5)

        plt.show()


    # Step 5 : Rapport & Conclusion --------------------------------------
    if report:
        print(f"=" * 70)
        print(f"📊 RAPPORT DE CORRÉLATION ENTRE '{var1}' ET '{var2}'")
        print("=" * 70)
        print(f"🔍 Type de corrélation utilisée : Chi2")
        print(f"📊 V de Cramer : {v_cramer:.4f}, corrélation {'faible' if v_cramer <= 0.1 else 'modérée' if 0.1 < v_cramer <= 0.3 else 'forte'}")
        print(f"📊 P-value : {p_value:.4f}")

        if p_value < seuil:
            print(f"\n✅ CONCLUSION: Il existe une corrélation statistiquement significative entre '{var1}' et '{var2}' (p < {seuil}).")
        else:
            print(f"\n⚠️ CONCLUSION : Il n'existe pas de corrélation statistiquement significative entre '{var1}' et '{var2}' (p >= {seuil}).")



def corr_quanti_quali(df, var1, var2, seuil=0.05, plot=True, report=True):
    """
    Analyse la corrélation entre une variable quantitative et une variable qualitative.
    Sélectionne automatiquement le test selon la normalité et le nombre de groupes :
    - 2 groupes, normal    → Test T + Cohen's d
    - 2 groupes, non-param → Mann-Whitney
    - 3+ groupes, normal   → ANOVA + Levene + Eta²
    - 3+ groupes, non-param → Kruskal-Wallis

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        var1 (str): Nom de la première variable (quanti ou quali, ordre indifférent).
        var2 (str): Nom de la deuxième variable (quanti ou quali, ordre indifférent).
        seuil (float): Seuil de significativité (par défaut 0.05).
        plot (bool): Affiche les graphiques si True (par défaut True).
        report (bool): Affiche le rapport final si True (par défaut True).

    Returns:
        None: Affiche les résultats dans la console.

    Example:
        >>> corr_quanti_quali(df, 'montant', 'categorie')
        >>> corr_quanti_quali(df, 'categorie', 'montant', plot=False)
    """

    # Initialisation de la fonction --------------------------------------
    print("=" * 70)
    print(f"📊 ANALYSE DE CORRÉLATION")
    print("=" * 70)


    # Step 1 : vérification préliminaire des données --------------------------------------
    print("\n" + "-" * 70)
    verif = verification_donnees(df, var1, var2)
    print(f"\n🔍 Analyse de la corrélation entre '{var1}' (type : {verif['dtype_var1']}) et '{var2}' (type : {verif['dtype_var2']})")
    liste_var = [var1, var2]
    liste_quanti = ['int64', 'float64']
    liste_quali = ['object', 'category']
    ## Détection de la var quanti et de la var quali
    var_quanti = None
    var_quali = None
    for var in liste_var:
        if df[var].dtype in liste_quanti:
            var_quanti = df[var]
        elif df[var].dtype in liste_quali:
            var_quali = df[var]

    if verif['type_df'] == 'DataFrame' and var_quanti is not None and var_quali is not None and verif['shape'][0] > 2:
        print(f"✅ Vérification des données : OK, les variables sont quantitatives et qualitatives et le dataset est suffisant pour l'analyse.")
    else:
        print(f"❌ Vérification des données : Les variables doivent être quantitatives (int64 ou float64) qualitatives (object ou category) et le dataset doit contenir plus de 2 lignes pour effectuer une analyse de corrélation.")
        return
    

    # Step 2 : Vérification de la normalité des variables --------------------------------------
    print("\n" + "-" * 70)
    print(f"\n🔍 Vérification de la normalité de la variable '{var_quanti.name}'")
    normalite_var1 = verification_normalite(df, var_quanti.name)
    
    # Affichage des résultats de normalité et graphique pour la variable 1
    print(f"P-value pour {normalite_var1['test']} '{var_quanti.name}' : {normalite_var1['p_value']:.4f}")
    if plot:
        plot_normalite(df, [var_quanti.name])
    
    if normalite_var1['p_value'] > seuil:
        print(f"✅ La variable suit une distribution normale (p > {seuil}).")
        normalite = True
    else:
        print(f"⚠️ La variable ne suit pas une distribution normale (p <= {seuil}).")
        normalite = False


    # Step 3 : Calcul de la corrélation --------------------------------------
    print("\n" + "-" * 70)
    n_groupes = var_quali.nunique()
    groupes = []
    for unique in var_quali.unique():
        groupes.append(df[df[var_quali.name] == unique][var_quanti.name])
    if n_groupes <= 2:
        if normalite:
            # Test T
            stat, p_value = ttest_ind(groupes[0], groupes[1])
            cohen = cohen_d(groupes[0], groupes[1])
        else:
            # Mann Whitney (Test U)
            stat, p_value = mannwhitneyu(groupes[0], groupes[1])
    else:
        if normalite:
            # ANOVA (f_oneway)
            stat, p_value = f_oneway(*groupes)
            stat_levene, p_value_levene = levene(*groupes) # Levene pour affichage uniquement
            eta_carre = eta_squared(groupes)
        else:
            # Kruskal-Wallis
            stat, p_value = kruskal(*groupes)
    
    if p_value < seuil:
        print(f"\n✅ La corrélation est statistiquement significative (p < {seuil}).")
    else:
        print(f"\n⚠️ La corrélation n'est pas statistiquement significative (p >= {seuil}).")


    # Step 4 : Visualisation de la corrélation --------------------------------------
    if plot: 
        # Boxplot
        fig = px.box(
            df,
            x=var_quali.name,
            y=var_quanti.name,
            title=f'Distribution conjointe : {var1} et {var2}',
            labels={var1: f'{var1}', var2: f'{var2}'},
        )

        fig.update_layout(template='plotly_white', height=600)
        fig.show()

    
    # Step 5 : Rapport & Conclusion --------------------------------------
    if report:
        print(f"=" * 70)
        print(f"📊 RAPPORT DE CORRÉLATION ENTRE '{var1}' ET '{var2}'")
        print("=" * 70)
        if n_groupes <= 2:
            if normalite:
                print(f"🔍 Type de corrélation utilisée : Test T")
                print(f"📊 Statistique t : {stat:.4f}")
                print(f"📊 P-value : {p_value:.4f}")
                print(f"📊 Cohen's d : {cohen:.2f} -> effet {'négligeable' if abs(cohen) < 0.2 else 'faible' if 0.2 <= abs(cohen) < 0.5 else 'modéré' if 0.5 <= abs(cohen) < 0.8 else 'fort'}")
            else:
                print(f"🔍 Type de corrélation utilisée : Mann-Whitney (Test U)")
                print(f"📊 Statistique U : {stat:.4f}")
                print(f"📊 P-value : {p_value:.4f}")
        else:
            if normalite:
                print(f"🔍 Type de corrélation utilisée : ANOVA")
                print(f"📊 Statistique F : {stat:.4f}")
                print(f"📊 P-value : {p_value:.4f}")
                print(f"📊 Statistique Levene : {stat_levene:.4f}")
                print(f"📊 P-value Levene : {p_value_levene:.4f}")
                print(f"📊 Eta² : {eta_carre:.2f} -> '{var_quali.name}' explique {eta_carre * 100:.0f}% de la variance de '{var_quanti.name}'")
            else:
                print(f"🔍 Type de corrélation utilisée : Kruskal-Wallis")
                print(f"📊 Statistique H : {stat:.4f}")
                print(f"📊 P-value : {p_value:.4f}")

        if p_value < seuil:
            print(f"\n✅ CONCLUSION: Il existe une corrélation statistiquement significative entre '{var1}' et '{var2}' (p < {seuil}).")
        else:
            print(f"\n⚠️ CONCLUSION : Il n'existe pas de corrélation statistiquement significative entre '{var1}' et '{var2}' (p >= {seuil}).")