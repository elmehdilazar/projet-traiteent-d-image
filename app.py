import cv2
import imutils as im

# Lisez le fichier image
input_file = 'car3.jpg'
image = cv2.imread(input_file)

# Vérifiez si l'image était chargée avec succès
if image is None:
    print("Error loading image file:", input_file)
    exit(1)

#Redimensionner l'image - Changer la largeur à 500
new_width = 500
image = im.resize(image, width=new_width)

# Convertir l'image RVB en niveaux de gris
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Appliquer un filtre bilatéral pour éliminer le bruit tout en préservant les bords
d = 11
sigma_color = 17
sigma_space = 17
filtered_img = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)

# Trouver des bords de l'image en niveaux de gris
lower_threshold = 170
upper_threshold = 200
edged = cv2.Canny(filtered_img, lower_threshold, upper_threshold)

# Trouver des contours en fonction des bords
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#Trier les contours par zone dans l'ordre descendant et sélectionnez les 10 meilleurs contours
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
number_plate_cnt = None
print("Number of Contours found:", len(contours))

#Boucle sur les contours pour trouver le meilleur contour approximatif de la plaque d'immatriculation
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.01 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:  # Sélectionnez le contour avec 4 coins
        print(approx)
        number_plate_cnt = approx  # Ceci est notre contour approximatif de la plaque d'immatriculation
        break

# Afficher l'image d'origine
cv2.imshow("Input Image", image)

# Afficher l'image en niveaux de gris
#cv2.imshow("Gray scale Image", gray)

# #Afficher l'image filtrée
#cv2.imshow("After Applying Bilateral Filter", filtered_img)

# # Afficher les bords émaux
#cv2.imshow("After Canny Edges", edged)

# Dessinez le contour sélectionné sur l'image d'origine
if number_plate_cnt is not None:
    cv2.drawContours(image, [number_plate_cnt], -1, (255, 0, 0), 2)

#Afficher la sortie finale
cv2.imshow("Output", image)

# Attendre la saisie de l'utilisateur avant de fermer les images affichées
cv2.waitKey(0)

# Fermez toutes les fenêtres
cv2.destroyAllWindows()
print("good by")