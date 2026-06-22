"""
Évaluation de modèles de classification.

Convention : la classe d'intérêt (celle qu'on veut détecter) doit être
encodée comme True/1 dans y. C'est elle qui sert de classe positive
pour toutes les métriques et la courbe ROC.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
)


def evaluate_model(model, X, y, target_names, methode_score, title="Matrice de confusion"):
    """
    Affiche la matrice de confusion, la courbe ROC, et calcule un score.

    Parameters
    ----------
    model : sklearn estimator
        Modèle entraîné, exposant predict() et predict_proba().
    X : array-like
        Données d'entrée (ex. X_test).
    y : array-like
        Labels réels (booléens), True = classe d'intérêt.
    target_names : list[str]
        Noms des classes dans l'ordre trié (ex. ["Vrai", "Faux"]).
    methode_score : str
        "accuracy", "precision", "recall" ou "f1".
    title : str, optional
        Titre du graphique de la matrice de confusion.

    Returns
    -------
    float
        Le score demandé.
    """
    y_predict = model.predict(X)

    # --- Matrice de confusion ---
    cm_pct = confusion_matrix(y, y_predict, normalize='true')
    cm_count = confusion_matrix(y, y_predict)

    counts = ["{0:0.0f}\n".format(value) for value in cm_count.flatten()]
    percentages = ["{0:.1%}".format(value) for value in cm_pct.flatten()]
    box_labels = [f"{v1}{v2}" for v1, v2 in zip(counts, percentages)]
    box_labels = np.asarray(box_labels).reshape(cm_pct.shape[0], cm_pct.shape[1])

    plt.figure(figsize=(8, 5), dpi=100)
    plt.title(title)
    sns.heatmap(cm_pct, vmin=0.0, vmax=1.0, cmap='Blues',
                annot=box_labels, fmt='',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Valeurs prédites")
    plt.ylabel("Valeurs réelles")
    plt.show()

    # --- Courbe ROC ---
    # predict_proba renvoie une colonne de probabilité par classe.
    # On retrouve l'index de la colonne correspondant à True (notre classe d'intérêt).
    y_proba = model.predict_proba(X)
    pos_idx = list(model.classes_).index(True)
    y_score = y_proba[:, pos_idx]  # probabilité d'être "True"

    fpr, tpr, _ = roc_curve(y, y_score)
    auc = roc_auc_score(y, y_score)

    plt.figure(figsize=(6, 6), dpi=100)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Hasard")
    plt.xlabel("Taux de faux positifs (FPR)")
    plt.ylabel("Taux de vrais positifs (TPR)")
    plt.title("Courbe ROC")
    plt.legend()
    plt.show()

    # --- Score demandé ---
    if methode_score == "accuracy":
        score = accuracy_score(y, y_predict)
    elif methode_score == "precision":
        score = precision_score(y, y_predict)
    elif methode_score == "recall":
        score = recall_score(y, y_predict)
    elif methode_score == "f1":
        score = f1_score(y, y_predict)
    else:
        raise ValueError("Erreur de saisie sur le nom de la methode_score : \naccuracy\nprecision\nrecall\nf1")

    print(f"{methode_score}_score : {100 * score:.2f}%")

    return score