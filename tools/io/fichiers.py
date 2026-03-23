import os

def ouvrir_fichier_local(filename, mode='r', encoding=None):
    """
    Ouvre un fichier situé dans le même dossier que ce script.
    
    Args:
        filename (str): Nom du fichier à ouvrir.
        mode (str): Mode d'ouverture (par défaut 'r').
        encoding (str, optionnel): Encodage à utiliser (ex: 'utf-8').
    
    Returns:
        file object: Fichier ouvert prêt à être lu ou écrit.
    """
    chemin_script = os.path.dirname(os.path.abspath(__file__))
    chemin_fichier = os.path.join(chemin_script, filename)
    if encoding:
        return open(chemin_fichier, mode, encoding=encoding)
    else:
        return open(chemin_fichier, mode)
    

def get_nom_fichier(filename):
    """
    Récupère le nom d'un fichier sans son extension.
    
    Args:
        filename (str): Nom du fichier à ouvrir.
        
    Returns:
        nom_entree: Fichier sans l'extension.
    """
    nom_entree = os.path.splitext(filename)[0].lower()
    return nom_entree


def get_extension_fichier(filename):
    """
    Récupère l'extension d'un fichier.
    
    Args:
        filename (str): Nom du fichier à ouvrir.
        
    Returns:
        filetype (str): Extension.
    """
    filetype = os.path.splitext(filename)[1].lower()
    return filetype