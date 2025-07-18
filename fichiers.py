import os

def ouvrir_fichier_local(nom_fichier, mode='r', encoding=None):
    """
    Ouvre un fichier situé dans le même dossier que ce script.
    
    Args:
        nom_fichier (str): nom du fichier à ouvrir.
        mode (str): mode d'ouverture (par défaut 'r').
        encoding (str, optionnel): encodage à utiliser (ex: 'utf-8').
    
    Returns:
        file object: fichier ouvert prêt à être lu ou écrit.
    """
    chemin_script = os.path.dirname(os.path.abspath(__file__))
    chemin_fichier = os.path.join(chemin_script, nom_fichier)
    if encoding:
        return open(chemin_fichier, mode, encoding=encoding)
    else:
        return open(chemin_fichier, mode)