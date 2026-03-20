# ------------------------
# Créer un fichier de logs
# ------------------------

import logging
# Configure le journal (va créer un fichier log.txt)
logging.basicConfig(
    filename=r'chemin\log.txt', 
    level=logging.INFO, # gere le niveau d'ecriture (debug, info, warning, error, critical)
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8')