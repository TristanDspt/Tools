def verification_donnees(df, var1, var2):
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
        'dtype_var2': df[var2].dtype if var2 in df.columns else None,
        'shape': df.shape
    }
    
    return resultats