# Arborescence des tests statistiques

---

## 🎯 Usages courants — Data Analyst & Dev IA

### En Data Analyse (quotidien)

| Situation | Test à utiliser | Corrélation |
|---|---|---|
| Explorer le lien entre deux variables numériques | Pearson (normal) ou Spearman (non-param.) | Quanti / Quanti
| Comparer deux groupes sur une mesure | Test T (normal) ou Mann-Whitney (non-param.) | Quanti / Quali
| Croiser deux variables catégorielles | Chi² + V de Cramér | Quali / Quali
| Comparer plusieurs groupes sur une mesure | ANOVA (normal) ou Kruskal-Wallis (non-param.) | Quanti / Quali
| Vérifier si les variances sont comparables | Levene | Pré requis ANOVA

### En Dev IA / ML (formation et projets)

| Situation | Test à utiliser |
|---|---|
| Comparer les perfs de deux modèles | Test T apparié ou Wilcoxon |
| Vérifier qu'une feature est utile (régression) | F-test / p-value des coefficients |
| Mesurer la corrélation entre features | Pearson / Spearman |
| Évaluer l'association feature × cible catégorielle | Chi² |
| Tester la normalité des résidus d'un modèle | Shapiro-Wilk |

> **En pratique** : Pearson/Spearman, Chi², Test T/Mann-Whitney et ANOVA couvrent 90% des besoins quotidiens d'un Data Analyst. Le reste se googlee quand le besoin arrive.

---

## 1. Quanti / Quanti

```mermaid
flowchart TD
    A["Quanti / Quanti"] --> B{"Distribution normale ?<br/>— Shapiro-Wilk"}

    B -->|Oui — paramétrique| C["Pearson<br/>— mesure d'effet : R²"]
    B -->|Non — non-paramétrique| D["Spearman<br/>— mesure d'effet : R² de Spearman"]

    C -->|Vérifier avant modélisation| E["Levene<br/>— égalité des variances"]
    E -->|Variances homogènes| F["Régression linéaire<br/>— mesure d'effet : R², F-test global"]

    B -->|Comparer deux distributions empiriques| G["Kolmogorov-Smirnov<br/>— alternatif à Mann-Whitney<br/>— compare la forme entière<br/>— limité à 2 groupes"]

    classDef test fill:#0d6e56,color:#9fe1cb,stroke:#0f6e56
    classDef question fill:#444441,color:#d3d1c7,stroke:#5f5e5a
    classDef side fill:#854f0b,color:#fac775,stroke:#ba7517
    class C,D,E,F test
    class A,B question
    class G side
```

> **Régression linéaire** : suppose la normalité (même chemin que Pearson) + homoscédasticité (Levene). Si Spearman → pas de régression linéaire classique.
>
> **KS** : alternative à Mann-Whitney pour 2 groupes quand on veut comparer la forme complète des distributions, pas seulement les médianes. Moins courant en pratique.

---

## 2. Quanti / Quali

```mermaid
flowchart TD
    A["Quanti / Quali"] --> B{"Nombre de groupes ?"}

    B -->|2 groupes| C{"Distribution normale ?<br/>— Shapiro-Wilk"}
    C -->|Oui| D["Test T — Student<br/>— compare les moyennes<br/>— mesure d'effet : Cohen's d"]
    C -->|Non| E["Mann-Whitney<br/>— compare les rangs/médianes<br/>— non-paramétrique"]
    C -->|Non, comparer distributions| KS["Kolmogorov-Smirnov<br/>— compare la forme entière"]

    B -->|3 groupes ou plus| F{"Distribution normale ?<br/>— Shapiro-Wilk"}
    F -->|Oui| G{"Variances égales ?<br/>— Levene"}
    G -->|Oui| H["ANOVA — F-test<br/>— mesure d'effet : Eta²"]
    G -->|Non| I["Welch ANOVA<br/>— robuste à l'hétéroscédasticité"]
    F -->|Non| J["Kruskal-Wallis<br/>— compare les rangs<br/>— non-paramétrique"]

    classDef test fill:#533ab7,color:#cecbf6,stroke:#534ab7
    classDef question fill:#444441,color:#d3d1c7,stroke:#5f5e5a
    classDef side fill:#854f0b,color:#fac775,stroke:#ba7517
    class D,E,H,I,J test
    class A,B,C,F,G question
    class KS side
```

> **Mann-Whitney ≠ Test T** : Mann-Whitney compare les rangs (médianes), pas les moyennes. Alternative non-paramétrique, pas un substitut exact.
>
> **Levene > Fisher** comme pré-requis ANOVA : Fisher est trop sensible à la non-normalité pour être fiable dans ce rôle.
>
> **KS** : placé ici en parallèle de Mann-Whitney pour 2 groupes (logique souvent enseignée), mais compare la forme des distributions plutôt que les médianes.

---

## 3. Quali / Quali

```mermaid
flowchart TD
    A["Quali / Quali"] --> B{"Que veux-tu faire ?"}

    B -->|Tester l'existence d'un lien| C["Test du Chi²<br/>— y a-t-il une association ?"]
    C -->|Si lien significatif, mesurer sa force| D["V de Cramér<br/>— mesure d'effet du Chi²"]

    B -->|Modéliser / prédire| E{"Modalités de la cible ?"}
    E -->|2 modalités| F["Régression logistique binaire"]
    E -->|3+ modalités non ordonnées| G["Régression logistique multinomiale"]
    E -->|3+ modalités ordonnées| H["Régression ordinale"]

    classDef test fill:#993c1d,color:#f5c4b3,stroke:#993c1d
    classDef question fill:#444441,color:#d3d1c7,stroke:#5f5e5a
    class C,D,F,G,H test
    class A,B,E question
```

> **Chi² → V de Cramér** : deux étapes séquentielles. D'abord Chi² (est-ce qu'il y a un lien ?), ensuite V de Cramér (c'est fort comment ?). Pas deux chemins alternatifs.

---

## Récap global

| Cas | Test | Mesure d'effet |
|---|---|---|
| Quanti/Quanti — corrélation normale | Pearson | R² |
| Quanti/Quanti — corrélation non-param. | Spearman | R² de Spearman |
| Quanti/Quanti — égalité des variances | Levene | — |
| Quanti/Quanti — comparer distributions | KS | — |
| Quanti/Quanti — modélisation | Régression linéaire | R², F-test |
| Quanti/Quali — 2 groupes, normal | Student (T) | Cohen's d |
| Quanti/Quali — 2 groupes, non-param. | Mann-Whitney (U) | — |
| Quanti/Quali — 2 groupes, distributions | KS | — |
| Quanti/Quali — 3+ groupes, normal | ANOVA / Welch ANOVA | Eta² |
| Quanti/Quali — 3+ groupes, non-param. | Kruskal-Wallis | — |
| Quali/Quali — association | Chi² | V de Cramér |
| Quali/Quali — modélisation binaire | Régression logistique | — |
| Quali/Quali — modélisation multi | Régression multinomiale / ordinale | — |

> **Test ≠ mesure d'effet** : le test dit *"c'est significatif ?"*, la mesure d'effet dit *"c'est fort ?"*
>
> **Pas de vérité absolue** : le choix du test dépend aussi du contexte, du domaine, et des conventions de ton équipe. Ce fichier est une boussole, pas une loi.
