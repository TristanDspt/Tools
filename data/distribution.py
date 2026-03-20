import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import verification_donnees

def lorenz(df, var1, titre_x, titre_y):
    """
    Trace la courbe de Lorenz et calcule l'indice de Gini pour une variable quantitative.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        var1 (str): Nom de la colonne quantitative à analyser (valeurs > 0).
        titre_x (str): Label de l'axe X (ex: "des clients", "des produits").
        titre_y (str): Label de l'axe Y (ex: "des revenus", "des prix").
        
    Returns:
        None: Affiche le graphe et les résultats dans la console.
        
    Example:
        >>> lorenz(df, "price", "des produits", "des prix")
    """

    # Initialisation de la fonction
    print("=" * 70)
    print(f"📊 COURBE DE LORENZ")
    print("=" * 70)

    # Vérification des données
    print("\n" + "-" * 70)
    verif = verification_donnees(df=df, var1=var1)
    print(f"🔍 Analyse de la variable '{var1}' (type : {verif['dtype_var1']})")
    
    print(f"📊 Type du dataset : {verif['type_df']}")
    print(f"📊 Forme du DataFrame : {verif['shape']}")
    
    if verif['type_df'] == 'DataFrame' and verif['dtype_var1'] in ['int64', 'float64'] and verif['shape'][0] > 2:
        print(f"✅ Vérification des données : OK, la variable est quantitatives et le dataset est suffisant pour l'analyse.")
    else:
        print(f"❌ Vérification des données : La variables doit être quantitatives (int64 ou float64) et le dataset doit contenir plus de 2 lignes.")
        return

    # Lorenz
    valeurs = df[df[var1] > 0]
    valeurs_sorted = np.sort(valeurs[var1])

    lorenz = np.cumsum(valeurs_sorted) / valeurs_sorted.sum()
    lorenz = np.insert(lorenz, 0, 0)

    xaxis = np.linspace(0, 1, len(lorenz))
    plt.plot(xaxis, lorenz, drawstyle='default', color="#04339A")
    plt.plot([0, 1], [0, 1], color='#800020', linestyle='--')
    plt.xlabel(f"Part cumulée {titre_x}", fontsize=12)
    plt.ylabel(f"Part cumulée {titre_y}", fontsize=12)

    # Gini
    auc = np.trapezoid(lorenz, xaxis)
    gini = 1 - 2 * auc

    plt.title(f"Courbe de Lorenz (Indice de Gini : {gini:.2f})", fontname='Arial' , fontweight='bold', pad=10, fontsize=16.5, x=0.45, ha='center')
    plt.tick_params(axis='x', labelsize=11)
    plt.tick_params(axis='y', labelsize=11)