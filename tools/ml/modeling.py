"""
Utilitaires de modélisation pour la classification binaire.

Deux fonctionnalités principales :
- `custom_pipeline` : construit un Pipeline sklearn avec les étapes optionnelles
  (imputation, mise à l'échelle, réduction de dimension) avant le modèle.
- `evaluate_model` : affiche la matrice de confusion et la courbe ROC,
  puis renvoie le score choisi.

Convention : la classe d'intérêt (celle qu'on veut détecter) doit être
encodée comme True/1 dans y. C'est elle qui sert de classe positive
pour toutes les métriques et la courbe ROC.
"""

# Données et visualisation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning - Partitions
from sklearn.model_selection import train_test_split

# Machine Learning - Preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler # Standardisation des données numériques
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder # Encodage des données catégorielles
from sklearn.impute import SimpleImputer, KNNImputer # Imputation des valeurs manquantes

# Machine Learning - Pipeline
from sklearn.pipeline import Pipeline

# Modèles de classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Machine Learning - Evaluation  des modèles
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    classification_report, 
    ConfusionMatrixDisplay, 
    RocCurveDisplay
)

# Machine Learning - Optimisation des hyperparamètres
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV




def evaluate_model(model, X, y, target_names, methode_score, title="Matrice de confusion", roc=True):
    """
    Affiche la matrice de confusion et (optionnellement) la courbe ROC,
    puis renvoie le score demandé.

    Parameters
    ----------
    model : sklearn estimator
        Modèle entraîné exposant ``predict()`` et, si ``roc=True``,
        ``predict_proba()``.
    X : array-like de forme (n_samples, n_features)
        Données d'entrée (typiquement X_test).
    y : array-like de forme (n_samples,)
        Labels réels binaires ; True/1 = classe d'intérêt (classe positive).
    target_names : list[str]
        Noms des classes dans l'ordre croissant des labels
        (ex. ``[False, True]`` → ``["Négatif", "Positif"]``).
    methode_score : {"accuracy", "precision", "recall", "f1"}
        Métrique à calculer et à afficher.
    title : str, optional
        Titre du graphique de la matrice de confusion.
        Défaut : ``"Matrice de confusion"``.
    roc : bool, optional
        Si True (défaut), affiche également la courbe ROC avec l'AUC.

    Returns
    -------
    float
        Valeur du score sélectionné par ``methode_score``.

    Raises
    ------
    ValueError
        Si ``methode_score`` ne fait pas partie des valeurs autorisées.
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
    if roc:
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


def custom_pipeline(model, imputer=None, scaler=None, reducer=None):
    """
    Construit un Pipeline sklearn avec les étapes de prétraitement optionnelles.

    Les étapes sont ajoutées dans cet ordre si elles sont fournies :
    mise à l'échelle → imputation → réduction de dimension → modèle.

    .. warning::
        L'ordre scaler → imputer est intentionnel et important lorsque
        ``KNNImputer`` est utilisé : le KNN calcule des distances entre
        observations pour estimer les valeurs manquantes. Sans mise à
        l'échelle préalable, les variables avec les plus grandes valeurs
        absolues écrasent les autres dans ce calcul, ce qui biaise le
        choix des voisins et donc la valeur imputée.
        ``StandardScaler`` est utilisé par défaut pour cette raison.
        Si vous remplacez le scaler par ``None``, assurez-vous que vos
        données sont déjà normalisées avant d'appeler le pipeline.

    Parameters
    ----------
    model : sklearn estimator
        Modèle de classification ou de régression (étape finale du pipeline).
    imputer : sklearn transformer, optional
        Imputer pour les valeurs manquantes (ex. ``KNNImputer``, ``SimpleImputer``).
    scaler : sklearn transformer or None, optional
        Scaler pour la normalisation. Défaut : ``StandardScaler()``.
        Passer ``None`` pour désactiver la mise à l'échelle.
    reducer : sklearn transformer, optional
        Réducteur de dimension (ex. ``PCA``).

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline prêt à être entraîné avec ``fit()`` / ``fit_transform()``.
    """
    if scaler is None:
        scaler = StandardScaler()
    steps = []
    if scaler is not None:
        steps.append(('scaler', scaler))
    if imputer is not None:
        steps.append(('imputation', imputer))
    if reducer is not None:
        steps.append(('reduction', reducer))
    steps.append(('model', model))
    return Pipeline(steps)