from collections import Counter
import requests
import random
import math

def load_text_from_web(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"An error occurred while loading the text: {e}")
        return None

def cut_string_into_pairs(text):
    pairs = []
    for i in range(0, len(text) - 1, 2):
        pairs.append(text[i:i + 2])
    if len(text) % 2 != 0:
        pairs.append(text[-1] + '_')  # Add a placeholder if the string has an odd number of characters
    return pairs

# Charger les textes du Projet Gutenberg
url1 = "https://www.gutenberg.org/ebooks/13846.txt.utf-8"
url2 = "https://www.gutenberg.org/ebooks/4650.txt.utf-8"

text1 = load_text_from_web(url1)
text2 = load_text_from_web(url2)

if text1 is not None and text2 is not None:
    text = text1 + text2
else:
    text = ""

# Calculer la fréquence des caractères et des bicarractères
caracteres = list(text)
bicaracteres = cut_string_into_pairs(text)

frequence_caracteres = Counter(caracteres)
frequence_bicaracteres = Counter(bicaracteres)

# Combiner les fréquences des caractères et bicarractères
frequence_symboles = frequence_caracteres + frequence_bicaracteres

# Sélectionner les 256 symboles les plus fréquents
frequence_symboles_256 = Counter(frequence_symboles).most_common(256)
symboles_retenus = {symbole: count for symbole, count in frequence_symboles_256}

# Regrouper les symboles non retenus en utilisant un symbole générique basé sur la similarité
symboles_similaires = {'é': 'e', 'è': 'e', 'ê': 'e', 'à': 'a', 'â': 'a', 'ù': 'u', 'û': 'u', 'î': 'i', 'ï': 'i', 'ç': 'c'}
texte_simplifie = ""
for char in text:
    if char in symboles_retenus:
        texte_simplifie += char
    elif char in symboles_similaires:
        texte_simplifie += symboles_similaires[char]
    else:
        texte_simplifie += '_'  # Remplacer les autres symboles par '_'

# Normaliser les fréquences pour obtenir des probabilités
nb_symboles_total = sum(symboles_retenus.values())

probabilite_symboles = {symbole: count / nb_symboles_total for symbole, count in symboles_retenus.items()}

# Afficher quelques résultats
print("Probabilités des symboles les plus fréquents (limités à 256) :")
for symbole, prob in Counter(probabilite_symboles).most_common(20):
    print(f"{symbole}: {prob:.4f}")

# Sauvegarder les probabilités dans un fichier pour une utilisation ultérieure
with open("frequences_symboles_256.txt", "w", encoding="utf-8") as f:
    for symbole, prob in probabilite_symboles.items():
        f.write(f"{symbole}: {prob:.6f}\n")

# Sauvegarder le texte simplifié dans un fichier
with open("texte_simplifie.txt", "w", encoding="utf-8") as f:
    f.write(texte_simplifie)

# Générer une clé aléatoire pour chiffrer le texte

def gen_key(symboles):
    l = len(symboles)
    if l > 256:
        return False

    random.seed(1337)
    int_keys = random.sample(list(range(l)), l)
    dictionary = dict({})
    for s, k in zip(symboles, int_keys):
        dictionary[s] = "{:08b}".format(k)
    return dictionary

# Générer la clé de chiffrement
key = gen_key(list(symboles_retenus.keys()))

# Fonction pour convertir un message en symboles

def M_vers_symboles(M, K):
    encoded_text = []
    i = 0

    while i < len(M):
        # Vérifie les paires de caractères
        if i + 1 < len(M):
            pair = M[i] + M[i + 1]
            if pair in K:
                encoded_text.append(pair)
                i += 2  # Sauter les deux caractères utilisés
                continue

        # Vérifie le caractère seul
        if M[i] in K:
            encoded_text.append(M[i])
        else:
            # Conserve le caractère tel quel si non trouvé
            encoded_text.append(M[i])
        i += 1

    return encoded_text

# Fonction pour chiffrer le texte

def chiffrer(M, K):
    l = M_vers_symboles(M, K)
    l = [K[x] for x in l]
    return ''.join(l)

# Chiffrer un exemple de texte simplifié
M = texte_simplifie[:500]  # Prendre les 500 premiers caractères du texte simplifié
C = chiffrer(M, key)

print("Texte original simplifié :")
print(M)
print("\nTexte chiffré :")
print(C)

# Fonction d'évaluation de la qualité du texte déchiffré
def score_dechiffrement(texte):
    bigrammes = cut_string_into_pairs(texte)
    score = 0
    # Ajouter les fréquences des caractères individuels
    for char in texte:
        if char in probabilite_symboles:
            score += math.log(probabilite_symboles[char] + 1e-9)  # Ajouter une petite valeur pour éviter log(0)
    # Ajouter les fréquences des bigrammes
    for bigramme in bigrammes:
        if bigramme in probabilite_symboles:
            score += math.log(probabilite_symboles[bigramme] + 1e-9)  # Ajouter une petite valeur pour éviter log(0)
    return score

# Déchiffrement itératif pour améliorer la clé de substitution
def dechiffrement_iteratif(texte_chiffre, iterations=1000):
    # Générer une clé initiale aléatoire
    cle_actuelle = gen_key(list(symboles_retenus.keys()))
    meilleur_score = float('-inf')
    meilleure_cle = cle_actuelle.copy()

    for iteration in range(iterations):
        # Déchiffrer le texte avec la clé actuelle
        texte_dechiffre_actuel = analyse_de_frequence(texte_chiffre, cle_actuelle)
        score_actuel = score_dechiffrement(texte_dechiffre_actuel)

        # Perturber légèrement la clé pour créer une nouvelle clé
        nouvelle_cle = cle_actuelle.copy()
        k1, k2 = random.sample(list(nouvelle_cle.keys()), 2)
        nouvelle_cle[k1], nouvelle_cle[k2] = nouvelle_cle[k2], nouvelle_cle[k1]

        # Déchiffrer le texte avec la nouvelle clé
        texte_dechiffre_nouveau = analyse_de_frequence(texte_chiffre, nouvelle_cle)
        score_nouveau = score_dechiffrement(texte_dechiffre_nouveau)

        # Mettre à jour la clé si la nouvelle version est meilleure
        if score_nouveau > score_actuel:
            cle_actuelle = nouvelle_cle
            score_actuel = score_nouveau

        # Mettre à jour la meilleure clé si nécessaire
        if score_actuel > meilleur_score:
            meilleure_cle = cle_actuelle
            meilleur_score = score_actuel

    return meilleure_cle

# Modifier analyse_de_frequence pour utiliser la clé de substitution spécifique
def analyse_de_frequence(texte_chiffre, cle):
    # Diviser le texte chiffré en séquences de 8 bits
    symboles_chiffres = [texte_chiffre[i:i + 8] for i in range(0, len(texte_chiffre), 8)]
    texte_dechiffre = ""
    for symbole in symboles_chiffres:
        for k, v in cle.items():
            if v == symbole:
                texte_dechiffre += k
                break
    return texte_dechiffre

# Appliquer le déchiffrement itératif pour améliorer la clé de déchiffrement
meilleure_cle = dechiffrement_iteratif(C)

# Déchiffrer le texte chiffré avec la meilleure clé trouvée
texte_dechiffre_optimise = analyse_de_frequence(C, meilleure_cle)

print("\nTexte déchiffré optimisé :")
print(texte_dechiffre_optimise)

