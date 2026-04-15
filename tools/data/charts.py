import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt


# ── Config globale ──────────────────────────────────────────────────────────────
TEMPLATE = 'plotly_white'
HEIGHT   = 500

# Palettes dispo (référence rapide) :
# Catégorielles : px.colors.qualitative.Plotly / D3 / G10 / T10 / Dark24
# Séquentielles : 'Blues', 'Viridis', 'Plasma', 'Cividis', 'Turbo'
# Divergentes   : 'RdBu', 'RdYlGn', 'Spectral'


# ── Bar ────────────────────────────────────────────────────────────────────────
def bar(df, x, y, color=None, titre=None, horizontal=False, text_auto=True, palette=None, barmode='group'):
    """
    Trace un barplot.

    Args:
        df (pd.DataFrame): Données.
        x (str): Colonne axe X.
        y (str): Colonne axe Y.
        color (str): Colonne pour grouper les barres (optionnel).
        titre (str): Titre du graphique.
        horizontal (bool): Barres horizontales si True.
        text_auto (bool): Affiche les valeurs sur les barres.
        palette: Couleur unique '#2196F3', liste ['red','blue'], ou palette qualitative px.colors.qualitative.D3.
        barmode (str): Mode d'affichage des barres. 'group' (défaut) ou 'stack'.

    Returns:
        fig: Figure Plotly.

    Example:
        >>> bar(df, x='categorie', y='montant', titre='CA par catégorie')
        >>> bar(df, x='categorie', y='montant', color='genre', palette=px.colors.qualitative.D3)
        >>> bar(df, x='categorie', y='montant', palette='#2196F3')
    """
    orientation = 'h' if horizontal else 'v'
    fig = go.Figure()

    if color:
        groupes = df[color].unique()
        couleurs = palette if isinstance(palette, list) else px.colors.qualitative.Plotly
        for i, groupe in enumerate(groupes):
            mask = df[color] == groupe
            x_val = df[mask][y] if horizontal else df[mask][x]
            y_val = df[mask][x] if horizontal else df[mask][y]
            fig.add_trace(go.Bar(
                x=x_val, y=y_val,
                name=str(groupe),
                orientation=orientation,
                marker_color=couleurs[i % len(couleurs)],
                text=y_val if text_auto else None,
                textposition='outside',
                texttemplate='%{value:,.0f}',
                hovertemplate='%{y} : %{x:,.0f}<extra></extra>' if horizontal else '%{x} : %{y:,.0f}<extra></extra>',
            ))
    else:
        x_val = df[y] if horizontal else df[x]
        y_val = df[x] if horizontal else df[y]
        fig.add_trace(go.Bar(
            x=x_val, y=y_val,
            orientation=orientation,
            marker_color=(palette if isinstance(palette, str) 
                else (palette if len(palette) == len(df) 
                else palette[0] if isinstance(palette, list) 
                else '#636EFA')),
            text=y_val if text_auto else None,
            textposition='outside',
            texttemplate='%{value:,.0f}',
            hovertemplate='%{y} : %{x:,.0f}<extra></extra>' if horizontal else '%{x} : %{y:,.0f}<extra></extra>',
        ))

    fig.update_layout(
        title=titre, 
        template=TEMPLATE, 
        height=HEIGHT, 
        barmode=barmode, 
        separators=". ", 
        hovermode='y unified' if horizontal else 'x unified',
        )
    return fig


# ── Donut ───────────────────────────────────────────────────────────────────────
def donut(df, names, values, titre=None, hole=0.6, palette=None):
    """
    Trace un donut chart.

    Args:
        df (pd.DataFrame): Données.
        names (str): Colonne des libellés.
        values (str): Colonne des valeurs.
        titre (str): Titre du graphique.
        hole (float): Taille du trou (0 = camembert).
        palette: Liste de couleurs ou palette qualitative px.colors.qualitative.D3.

    Returns:
        fig: Figure Plotly.

    Example:
        >>> donut(df, names='categorie', values='montant', palette=px.colors.qualitative.D3)
    """
    couleurs = palette if isinstance(palette, list) else px.colors.qualitative.Plotly

    fig = go.Figure(go.Pie(
        labels=df[names],
        values=df[values],
        hole=hole,
        marker_colors=couleurs,
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='%{label} : %{value:,.0f}<extra></extra>',
    ))
    fig.update_layout(title=titre, template=TEMPLATE, height=HEIGHT, separators=". ")
    return fig


# ── Line ────────────────────────────────────────────────────────────────────────
def line(df, x, y, color=None, titre=None, markers=False, palette=None):
    """
    Trace un line chart.

    Args:
        df (pd.DataFrame): Données.
        x (str): Colonne axe X (souvent une date).
        y (str | list): Colonne(s) axe Y.
        color (str): Colonne pour plusieurs séries (optionnel).
        titre (str): Titre du graphique.
        markers (bool): Affiche les points sur la courbe.
        palette: Couleur unique, liste ou palette qualitative px.colors.qualitative.D3.

    Returns:
        fig: Figure Plotly.

    Example:
        >>> line(df, x='date', y='montant', palette='#e63946')
        >>> line(df, x='date', y=['montant', 'budget'], palette=['#2196F3', '#FF5722'])
    """
    mode = 'lines+markers' if markers else 'lines'
    couleurs = palette if isinstance(palette, list) else px.colors.qualitative.Plotly
    fig = go.Figure()

    if color:
        groupes = df[color].unique()
        for i, groupe in enumerate(groupes):
            mask = df[color] == groupe
            fig.add_trace(go.Scatter(
                x=df[mask][x], y=df[mask][y],
                mode=mode,
                name=str(groupe),
                line_color=couleurs[i % len(couleurs)],
                hovertemplate='%{x} : %{y:,.0f}<extra></extra>',
            ))
    elif isinstance(y, list):
        for i, col in enumerate(y):
            c = palette if isinstance(palette, str) else couleurs[i % len(couleurs)]
            fig.add_trace(go.Scatter(
                x=df[x], y=df[col],
                mode=mode,
                name=col,
                line_color=c,
                hovertemplate='%{x} : %{y:,.0f}<extra></extra>',
            ))
    else:
        c = palette if isinstance(palette, str) else couleurs[0]
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y],
            mode=mode,
            name=y,
            line_color=c,
            hovertemplate='%{x} : %{y:,.0f}<extra></extra>',
        ))

    fig.update_layout(title=titre, template=TEMPLATE, height=HEIGHT, hovermode='x unified', separators=". ")
    return fig


# ── Scatter (px — trendline OLS) ─────────────────────────────────────────────────
def scatter(df, x, y, color=None, size=None, titre=None, trendline=None, palette=None):
    """
    Trace un nuage de points. Reste en px pour le support natif du trendline OLS.

    Args:
        df (pd.DataFrame): Données.
        x (str): Colonne axe X.
        y (str): Colonne axe Y.
        color (str): Colonne pour la couleur (optionnel).
        size (str): Colonne pour la taille des points (optionnel).
        titre (str): Titre du graphique.
        trendline (str): 'ols' pour une droite de régression (optionnel).
        palette: Couleur unique, liste ou palette qualitative px.colors.qualitative.D3.

    Returns:
        fig: Figure Plotly.

    Example:
        >>> scatter(df, x='age', y='montant', trendline='ols', palette='#2196F3')
        >>> scatter(df, x='age', y='montant', color='categorie', palette=px.colors.qualitative.D3)
    """
    color_discrete = palette if isinstance(palette, list) or (palette and color) else None

    fig = px.scatter(
        df, x=x, y=y,
        color=color,
        size=size,
        title=titre,
        trendline=trendline,
        trendline_color_override='red',
        opacity=0.6,
        color_discrete_sequence=color_discrete,
        template=TEMPLATE,
        height=HEIGHT
    )
    if palette and not color and not isinstance(palette, list):
        fig.update_traces(marker_color=palette, selector=dict(mode='markers'))

    return fig


# ── Box ─────────────────────────────────────────────────────────────────────────
def box(df, x, y, color=None, titre=None, points='outliers', palette=None):
    """
    Trace un boxplot.

    Args:
        df (pd.DataFrame): Données.
        x (str): Colonne catégorielle (axe X).
        y (str): Colonne quantitative (axe Y).
        color (str): Colonne pour la couleur (optionnel).
        titre (str): Titre du graphique.
        points (str | bool): 'outliers', 'all', False.
        palette: Couleur unique, liste ou palette qualitative px.colors.qualitative.D3.

    Returns:
        fig: Figure Plotly.

    Example:
        >>> box(df, x='categorie', y='montant', palette=px.colors.qualitative.D3)
    """
    couleurs = palette if isinstance(palette, list) else px.colors.qualitative.Plotly
    fig = go.Figure()

    groupes = df[color].unique() if color else df[x].unique()
    col_groupe = color if color else x

    for i, groupe in enumerate(groupes):
        mask = df[col_groupe] == groupe
        fig.add_trace(go.Box(
            x=df[mask][x],
            y=df[mask][y],
            name=str(groupe),
            marker_color=couleurs[i % len(couleurs)],
            boxpoints=points
        ))

    fig.update_layout(title=titre, template=TEMPLATE, height=HEIGHT, boxmode='group')
    return fig


# ── Histogram ───────────────────────────────────────────────────────────────────
def histo(df, x, color=None, titre=None, bins=30, barmode='overlay', palette=None):
    """
    Trace un histogramme.

    Args:
        df (pd.DataFrame): Données.
        x (str): Colonne à distribuer.
        color (str): Colonne pour superposer des groupes (optionnel).
        titre (str): Titre du graphique.
        bins (int): Nombre de barres.
        barmode (str): 'overlay', 'group' ou 'stack'.
        palette: Couleur unique, liste ou palette qualitative px.colors.qualitative.D3.

    Returns:
        fig: Figure Plotly.

    Example:
        >>> histo(df, x='age', palette='#2196F3')
        >>> histo(df, x='montant', color='genre', palette=['#2196F3', '#FF5722'])
    """
    couleurs = palette if isinstance(palette, list) else px.colors.qualitative.Plotly
    fig = go.Figure()

    if color:
        groupes = df[color].unique()
        for i, groupe in enumerate(groupes):
            mask = df[color] == groupe
            fig.add_trace(go.Histogram(
                x=df[mask][x],
                name=str(groupe),
                nbinsx=bins,
                marker_color=couleurs[i % len(couleurs)],
                opacity=0.7,
                hovertemplate='%{x} : %{y:,.0f}<extra></extra>',
            ))
    else:
        c = palette if isinstance(palette, str) else couleurs[0]
        fig.add_trace(go.Histogram(
            x=df[x],
            nbinsx=bins,
            marker_color=c,
            opacity=0.7,
            hovertemplate='%{x} : %{y:,.0f}<extra></extra>',
        ))

    fig.update_layout(title=titre, template=TEMPLATE, height=HEIGHT, barmode=barmode, bargap=0.05, hovermode='x unified')
    return fig


# ── Heatmap (corrélation) ────────────────────────────────────────────────────────
def heatmap_corr(df, titre='Matrice de corrélation', palette='RdBu'):
    """
    Trace une heatmap de la matrice de corrélation du DataFrame (seaborn).
    Masque automatiquement le triangle supérieur pour la lisibilité.

    Args:
        df (pd.DataFrame): Données (colonnes numériques uniquement).
        titre (str): Titre du graphique.
        palette (str): Palette divergente ex: 'RdBu', 'RdYlGn', 'coolwarm'.

    Returns:
        None: Affiche le graphique.

    Example:
        >>> heatmap_corr(df[['age', 'montant', 'score']])
        >>> heatmap_corr(df[['age', 'montant', 'score']], palette='coolwarm')
    """
    corr = df.select_dtypes('number').corr().round(2)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap=palette,
        center=0,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title(titre, fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()
