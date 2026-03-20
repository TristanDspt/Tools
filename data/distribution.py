import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Lorenz
prix = df_merge.query("price > 0")
prix_sorted = np.sort(prix.price)

lorenz = np.cumsum(prix_sorted) / prix_sorted.sum()
lorenz = np.insert(lorenz, 0, 0)

xaxis = np.linspace(0, 1, len(lorenz))
plt.plot(xaxis, lorenz, drawstyle='default', color='#004D40')
plt.plot([0, 1], [0, 1], color='#800020', linestyle='--')
plt.xlabel("Part cumulée des produits", fontsize=12)
plt.ylabel("Part cumulée des prix", fontsize=12)

# Gini
auc = np.trapezoid(lorenz, xaxis)
gini = 1 - 2 * auc

plt.title(f"Courbe de Lorenz (Indice de Gini : {gini:.2f})", fontname='Arial' , fontweight='bold', pad=10, fontsize=16.5, x=0.45, ha='center')
plt.tick_params(axis='x', labelsize=11)
plt.tick_params(axis='y', labelsize=11)