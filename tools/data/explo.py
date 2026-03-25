import pandas as pd

def sherlock(df):
    """
    Affiche un résumé rapide d'un DataFrame : dimensions, doublons, types,
    valeurs manquantes et détection de clés primaires potentielles.

    Args:
        df (pd.DataFrame): Le DataFrame à analyser.

    Returns:
        None: Affiche les résultats dans la console.

    Example:
        >>> sherlock(df)
        -- 100 lignes | 5 colonnes | 0 lignes doublons --
    """
    # Header rapide
    print(f"{df.shape[0]} lignes | {df.shape[1]} colonnes | {df.duplicated().sum()} lignes doublons")
    print("-" * 50)
    
    # Construction d'un tableau récapitulatif
    stats = pd.DataFrame({
        'Type': df.dtypes,
        'Manquants': df.isna().sum(),
        'Manquants %': (df.isna().mean() * 100).round(2).astype(str) + '%',
        'Uniques': df.nunique()
    })
    
    print(stats)
    print("-" * 50)

    potentielles_pks = stats[(stats['Uniques'] == df.shape[0]) & (stats['Manquants'] == 0)].index.tolist()
    if potentielles_pks:
        print(f"Clé potentielle sur {', '.join(potentielles_pks)}")
    else:
        print("Pas de colonne éligible en PK")