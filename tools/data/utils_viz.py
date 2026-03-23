from scipy import stats
import matplotlib.pyplot as plt

def plot_normalite(df, liste_var):
    """
    Affiche pour chaque variable un histogramme et un Q-Q plot pour évaluer visuellement la normalité.

    Args:
        df (pd.DataFrame): DataFrame contenant les données.
        liste_var (list): Liste des noms de colonnes à analyser.

    Returns:
        None: Affiche les graphiques dans la console.

    Example:
        >>> plot_normalite(df, ['age', 'montant'])
    """
    for var in liste_var:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Distribution de la variable               
        axes[0].hist(df[var], bins=30, edgecolor='white', color='#2196F3', alpha=0.7)
        axes[0].axvline(df[var].mean(), color='red', linestyle='--', label=f"Moyenne = {df[var].mean():.1f}")
        axes[0].axvline(df[var].median(), color='orange', linestyle='--', label=f"Médiane = {df[var].median():.1f}")
        axes[0].set_title(f'Distribution de la variable {var}', fontsize=12, fontweight='bold')
        axes[0].set_xlabel(f'{var}')
        axes[0].set_ylabel('Fréquence')
        axes[0].legend()

        # Q-Q plot de la variable
        stats.probplot(df[var], dist="norm", plot=axes[1])
        axes[1].set_title(f'Q-Q Plot : {var}', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show() 