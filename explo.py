import pandas as pd

def inspecteur_gadget(df):
    # Header rapide
    print(f"--- {df.shape[0]} lignes | {df.shape[1]} colonnes | {df.duplicated().sum()} doublons ---")
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