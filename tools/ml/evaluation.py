"""
Évaluation de modèles de classification.

Fournit une fonction pour visualiser la matrice de confusion et afficher
l'accuracy d'un modèle scikit-learn sur un jeu de données.
"""

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def evaluate_model(model, X, y, target_names, title="Matrice de confusion"):
    """
    Affiche la matrice de confusion et l'accuracy d'un modèle de classification.

    Chaque cellule de la heatmap affiche le nombre d'observations ainsi que
    le taux normalisé par ligne (rappel par classe). L'exactitude globale est
    également affichée dans la console.

    Parameters
    ----------
    model : sklearn estimator
        Modèle entraîné exposant une méthode predict().
    X : array-like of shape (n_samples, n_features)
        Données d'entrée (ex. X_test).
    y : array-like of shape (n_samples,)
        Labels réels correspondants (ex. y_test).
    target_names : list[str]
        Noms des classes, dans l'ordre des labels encodés.
    title : str, optional
        Titre du graphique (défaut : "Matrice de confusion").

    Returns
    -------
    float
        Score d'exactitude (accuracy) entre 0.0 et 1.0.
    """
    y_predict = model.predict(X)
    cm_pct = confusion_matrix(y, y_predict, normalize='true')
    cm_count = confusion_matrix(y, y_predict)

    counts = ["{0:0.0f}\n".format(value) for value in cm_count.flatten()]
    percentages = ["{0:.2%}".format(value) for value in cm_pct.flatten()]

    box_labels = [f"{v1}{v2}" for v1, v2 in zip(counts, percentages)]
    box_labels = np.asarray(box_labels).reshape(cm_pct.shape[0],cm_pct.shape[1])

    plt.figure(figsize=(8,5), dpi=100)

    plt.title(title)

    sns.heatmap(cm_pct,
                vmin=0.0,
                vmax=1.0,
                cmap='Blues',
                annot=box_labels,
                fmt='',
                xticklabels=target_names,
                yticklabels=target_names)

    plt.xlabel("Valeurs prédites")
    plt.ylabel("Valeurs réelles")

    plt.show()

    accuracy_score_result = accuracy_score(y, y_predict)

    print("L'exactitude est de : ", round(100.0*accuracy_score_result, 2), "%")

    return accuracy_score_result