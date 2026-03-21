import pandas as pd
from scipy.stats import shapiro, normaltest

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