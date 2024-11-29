import cv2

image = cv2.imread('../Comparison-of-image-classification-object-detection-instance-and-semantic-segmentation.png')

# Afficher l'image dans une fenêtre
cv2.imshow('Image', image)

while True:
    # Attendre 1 milliseconde pour un événement clavier
    key = cv2.waitKey()
    print(key)

    # Si la touche 'q' est pressée, sortir de la boucle
    if key == ord('q'):
        break

# Fermer la fenêtre d'affichage
cv2.destroyAllWindows()
