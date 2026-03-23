import pandas as pd
import numpy as np
from scipy.stats import shapiro, normaltest
from tools.data.correlation import corr_quanti_quanti, corr_quali_quali, corr_quanti_quali

def verification_donnees(df, var1, var2=None):
    """
    Vérifie les données pour s'assurer de la bonne exécution des tests statistiques.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        var1 (str): Nom de la première variable.
        var2 (str): Nom de la deuxième variable.
        
    Returns:
        dict: Dictionnaire contenant les résultats de la vérification.
        - 'type_df': Type de DataFrame (pandas ou plotly)
        - 'dtype_var1': Data Type de la variable 1
        - 'dtype_var2': Data Type de la variable 2
        - 'shape': Forme du DataFrame
    """

    resultats = {
        'type_df': type(df).__name__,
        'dtype_var1': df[var1].dtype if var1 in df.columns else None,
        'dtype_var2': df[var2].dtype if var2 is not None and var2 in df.columns else None,
        'shape': df.shape
    }
    
    return resultats

def verification_normalite(df, var):
    """
    Sélectionne et applique le test de normalité le plus adapté selon la taille de l'échantillon.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        var (str): Nom de la variable à tester.

    Returns:
        dict: Dictionnaire contenant les résultats du test.
        - 'test' (str): Nom du test utilisé ('Shapiro' ou 'D'Agostino')
        - 'p_value' (float): P-value du test (< 0.05 = non normale)
    """
    if len(df[var]) < 5000:
        _, p_shapiro = shapiro(df[var])
        resultats = {'test': 'Shapiro', 'p_value': p_shapiro}
    else:
        _, p_dagostino = normaltest(df[var])
        resultats = {'test': 'D\'Agostino', 'p_value': p_dagostino}
    
    return resultats

def cohen_d(groupe1, groupe2):
    """
    Calcule le d de Cohen, mesure d'effet pour le Test T.
    
    Args:
        groupe1 (pd.Series): Valeurs du premier groupe.
        groupe2 (pd.Series): Valeurs du deuxième groupe.
        
    Returns:
        float: Valeur du d de Cohen.
        - < 0.2 : effet négligeable
        - 0.2 - 0.5 : effet faible
        - 0.5 - 0.8 : effet modéré
        - > 0.8 : effet fort
    """
    n1, n2 = len(groupe1), len(groupe2)
    var1, var2 = groupe1.var(ddof=1), groupe2.var(ddof=1)
    std_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (groupe1.mean() - groupe2.mean()) / std_pooled

def eta_squared(groupes):
    """
    Calcule l'Eta², mesure d'effet pour l'ANOVA.
    
    Args:
        groupes (list): Liste de pd.Series, une par groupe.
        
    Returns:
        float: Valeur de l'Eta² (entre 0 et 1).
        - < 0.01 : effet négligeable
        - 0.01 - 0.06 : effet faible
        - 0.06 - 0.14 : effet modéré
        - > 0.14 : effet fort
    """
    grand_mean = np.concatenate([g.values for g in groupes]).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groupes)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in groupes)
    return ss_between / ss_total


def corr_pilote(df, var1, var2, seuil=0.05, plot=True, report=True):
    """
    Fonction dispatcher qui analyse automatiquement la corrélation entre deux variables
    en sélectionnant le test approprié selon leurs types.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        var1 (str): Nom de la première variable.
        var2 (str): Nom de la deuxième variable.
        seuil (float): Seuil de significativité (par défaut 0.05).
        plot (bool): Affiche les graphiques si True (par défaut True).
        report (bool): Affiche le rapport final si True (par défaut True).

    Returns:
        None: Affiche les résultats dans la console.

    Example:
        >>> corr_pilote(df, 'age', 'montant')
        >>> corr_pilote(df, 'genre', 'categorie', seuil=0.01, plot=False)
        >>> corr_pilote(df, 'montant', 'categorie')
    """

    liste_quanti = ['int64', 'float64']
    liste_quali = ['object', 'category']

    famille_var1 = 'quanti' if df[var1].dtype in liste_quanti else 'quali' if df[var1].dtype in liste_quali else None
    famille_var2 = 'quanti' if df[var2].dtype in liste_quanti else 'quali' if df[var2].dtype in liste_quali else None

    familles = {famille_var1, famille_var2}

    if familles == {'quanti', 'quanti'}:
        corr_quanti_quanti(df, var1, var2, seuil, plot, report)
    elif familles == {'quali', 'quali'}:
        corr_quali_quali(df, var1, var2, seuil, plot, report)
    elif familles == {'quanti', 'quali'}:
        corr_quanti_quali(df, var1, var2, seuil, plot, report)
    else:
        print("❌ Erreur les variables ne sont ni quantitave ni qualitative")