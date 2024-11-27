from ultralytics import YOLO
import multiprocessing


if __name__ == '__main__': # Vérifie que le script est exécuté directement (et non importé).
    multiprocessing.freeze_support() # Assure la compatibilité avec Windows pour l'exécution de processus parallèles.

    # Crée un nouveau modèle YOLO à partir d'un fichier de configuration.
    #model = YOLO("yolov8n.yaml") 
    # Charge un modèle YOLO pré-entraîné à partir d'un fichier .pt
    model = YOLO("yolov8n.pt") 

    # Entraîne le modèle sur un jeu de données et pour 3 époques.
    #model.train(data="coco128.yaml", epochs=3)
    # Évalue le modèle sur un ensemble de validation pour obtenir des métriques.
    #metrics = model.val()
    # Effectue une détection d'objets sur une image externe et affiche et enregistre les résultats.
    results = model("https://ultralytics.com/images/bus.jpg",show=True, save=True)
    #path = model.export(format="onnx")
    



# Remarques:
# 1 -> La première étape (création du modèle) prépare l'architecture et la configuration.
# 2 -> La seconde étape (chargement d'un modèle pré-entraîné) remplace cette architecture vide par un modèle déjà formé avec des poids pré-existants, ce qui permet d'atteindre de meilleurs résultats plus rapidement.


'''
Une époque (ou epoch en anglais) dans le cadre de l'entraînement d'un modèle de machine learning représente une passée complète à travers l'ensemble de données d'entraînement. C'est-à-dire qu'une époque correspond au nombre de fois que le modèle voit chaque exemple d'entraînement et ajuste ses poids en fonction des erreurs (grâce à la rétropropagation).
model.train(data="coco128.yaml", epochs=3)

indique que le modèle sera entraîné sur les données spécifiées dans le fichier coco128.yaml pendant 3 époques. Cela signifie que le modèle verra chaque image du jeu de données trois fois et ajustera ses poids à chaque époque.

    1ère époque : Le modèle apprend à partir des données d'entraînement.
    2e époque : Le modèle applique les ajustements de la 1ère époque et continue à améliorer ses prédictions.
    3e époque : Le modèle continue à ajuster ses poids et à réduire l'erreur.

En résumé, une époque est simplement un passage complet sur l'ensemble de données d'entraînement, et l'entraînement sur plusieurs époques permet au modèle de mieux généraliser et de s'améliorer au fil du temps.
'''

