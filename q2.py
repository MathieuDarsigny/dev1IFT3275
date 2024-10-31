import math
import os
import random as rnd
import re
import numpy as np
import requests
from collections import Counter

def cut_string_into_pairs(text):
  pairs = []
  for i in range(0, len(text) - 1, 2):
    pairs.append(text[i:i + 2])
  if len(text) % 2 != 0:
    pairs.append(text[-1] + '_')  # Add a placeholder if the string has an odd number of characters
  return pairs

def load_text_from_web(url):
  try:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    return response.text
  except requests.exceptions.RequestException as e:
    print(f"An error occurred while loading the text: {e}")
    return None
  

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

def chiffrer(M,K):
  l = M_vers_symboles(M, K)
  l = [K[x] for x in l]
  return ''.join(l)


def chiffrer2(M, K) -> str:
    """
    Encode le texte en utilisant un dictionnaire personnalisé.

    :param text: Le texte à encoder
    :param custom_dict: Le dictionnaire de correspondances
    :return: Le texte encodé
    """
    encoded_text = []
    i = 0

    while i < len(M):
        # Vérifie les paires de caractères
        if i + 1 < len(M):
            pair = M[i] + M[i + 1]
            if pair in K:
                encoded_text.append(K[pair])
                i += 2  # Sauter les deux caractères utilisés
                continue

        # Vérifie le caractère seul
        if M[i] in K:
            encoded_text.append(K[M[i]])
        else:
            # Conserve le caractère tel quel si non trouvé
            encoded_text.append(M[i])
        i += 1

    return ''.join(encoded_text)

def gen_key(symboles):

  l=len(symboles)
  if l > 256:
    return False
  # int_keys de 0 à 255 sans permutations
  int_keys = list(range(256))
  dictionary = dict({})
  for s,k in zip(symboles,int_keys):
    dictionary[s]="{:08b}".format(k )
  return dictionary

def text_to_symbols(text):
    caracteres = list(set(list(text)))
    nb_caracteres = len(caracteres)
    nb_bicaracteres = 256-nb_caracteres
    bicaracteres = [item for item, _ in Counter(cut_string_into_pairs(text)).most_common(nb_bicaracteres)]
    symboles = caracteres + bicaracteres
    return symboles

def eval_score(text, french_words):
    # Convertir la liste de caractères en une chaîne
    text_string = ''.join(text)  # Créer une chaîne à partir de la liste de caractères
    # Séparer le texte en mots uniques
    words = set(text_string.split())
    # Scorer selon le nombre de mots français trouvés et leur longueur
    score = 0
    for word in words:
        # Si le mot contient des 0 ou 1, ne pas le compter
        if '0' in word or '1' in word:
            continue

        word = word.lower() # Convertir le mot en minuscules

        # Enlever les caractères spéciaux du mot (sauf traits-d'union)
        word = re.sub(r"[^a-z\-êâïçîéèàûùëô]", "", word)

        # Vérifier si le mot est dans le dictionnaire des mots français
        if word in french_words:
            score += len(word)  # Ajouter la longueur du mot au score (pour favoriser les mots plus longs)

    # Ajuster le score selon la ponctuation et les majuscules
    '''for i in range(len(text_string) - 2):
        if text_string[i] in {".", "!", "?", "…"}:
            # Devrait être suivi d'un espace et d'une majuscule
            if text_string[i+2].isalpha():
                if text_string[i+1] == " " and text_string[i+2].isupper():
                    score += 1
                else:
                    score -=2
        elif text_string[i] in {",", ";", ":"}:
            # Devrait être suivi d'un espace et d'une minuscule
            if text_string[i+2].isalpha():
                if text_string[i+1] == " " and text_string[i+2].isupper():
                    score += 1
                else:
                    score -=2'''
    return score

def decode_text(encoded_text, weighted_probabilities, french_words):
    # Sachant que le texte encodé est des 1 et des 0 où chaque tranche de 8 bits est un caractère

    # Déterminer le nombre de 0 à ajouter au début du texte pour obtenir un nombre de bits multiple de 8
    num_padding_bits = (8 - (len(encoded_text) % 8))%8

    # Ajouter les bits de padding au début du texte
    encoded_text = "0" * num_padding_bits + encoded_text

    # Séparer le texte encodé en tranches de 8 caractères
    encoded_bytes = [encoded_text[i:i + 8] for i in range(0, len(encoded_text), 8)]

    # Obtenir les probabilités de chaque séquence de 8 bits
    byte_probabilities = {byte: 0 for byte in set(encoded_bytes)}
    for byte in encoded_bytes:
        byte_probabilities[byte] += 1
    total_bytes = sum(byte_probabilities.values())
    byte_probabilities = {byte: count / total_bytes for byte, count in byte_probabilities.items()}

    # Trier les séquences de 8 bits en ordre décroissant de probabilité
    sorted_bytes = sorted(byte_probabilities, key=byte_probabilities.get, reverse=True)

    # Trier les caractères en ordre décroissant de probabilité
    sorted_chars = sorted(weighted_probabilities, key=weighted_probabilities.get, reverse=True)

    # ALGORITHME PRINCIPAL DE DÉCODAGE
    # La clé de décodage
    key = {byte: None for byte in byte_probabilities}

    decoded_text = encoded_bytes

    # Boucle principale de travail. On itère sur tous bytes décroissant de probabilité et on cherche le meilleur caractère
    index_bytes = 0 # Permet de savoir où on est rendu dans les substitutions des séquences de 8 bits triées
    decoded_text_cpy = decoded_text.copy()
    for i in range(0, len(sorted_bytes)):
        for char in sorted_chars:
            if not char in key.values(): # On ne veut pas remplacer un caractère déjà remplacé
                byte = sorted_bytes[i]
                # On essaie de remplacer le byte par le caractère
                key[byte] = char

                # Déchiffrer le message encodé en utilisant la clé actuelle
                for j in range(len(decoded_text_cpy)):
                    if decoded_text_cpy[j] == byte:
                        decoded_text_cpy[j] = char

                # Évaluer le message déchiffré avec le dictionnaire
                score = eval_score(decoded_text, french_words)

                # Optimiser la clé de décodage en faisant des permutations de substitutions
                key, decoded_text, score = optimize_key(key, weighted_probabilities, decoded_text, french_words, score)

    return ''.join(decoded_text)

def optimize_key(key, weighted_probabilities, decoded_text, french_words, score):
    # Le cout d'optimiser la clé de 1 à 256 caractères est de 2 800 000 appels à la fonction
    #     2 solutions:
    #         - Optimiser la fonction
    #         - Ignorer certains "swap"
    # Répéter jusqu'à ce qu'il n'y ait plus d'améliorations significatives:
    #   - Pour chaque paire de caractères de la clé, si les fréquences des caractères ne sont pas trop éloignées, 
    #     échanger les caractères
    #   - Évaluer le message déchiffré avec le dictionnaire
    #   - Si le score est meilleur, mettre à jour la clé
    best_score = score
    new_text = decoded_text.copy()
    recommencer = False
    for int1 in key:
        char1 = key[int1]
        if char1 is not None:
            for int2 in key:
                char2 = key[int2]
                if char2 is not None and not char1 == char2:
                    # Vérifier si les fréquences des caractères ne sont pas trop éloignées
                    prob1 = weighted_probabilities[char1]
                    prob2 = weighted_probabilities[char2]
                    # Si la différence des probabilités est plus grande que la plus petite probabilité, ne pas échanger
                    if abs(prob1 - prob2) > min(prob1, prob2):
                        continue

                    # Échanger les caractères dans new_text
                    for i in range(len(new_text)):
                        if new_text[i] == char1:
                            new_text[i] = char2
                        elif new_text[i] == char2:
                            new_text[i] = char1

                    # Évaluer le message déchiffré avec le dictionnaire
                    new_score = eval_score(decoded_text, french_words)

                    # Si le score est meilleur
                    if new_score > best_score:
                        # Mettre à jour le score
                        best_score = new_score

                        # Mettre à jour le texte déchiffré
                        decoded_text = new_text

                        # Échanger les caractères dans la clé
                        key[int1], key[int2] = char2, char1

                        # Recommencer la boucle externe puisqu'on a modifié la clé
                        recommencer = True
                        print(f"Échange de {char1} ({prob1}) et {char2} ({prob2})")
                        break
    if recommencer:
        return optimize_key(key, weighted_probabilities, decoded_text, french_words, best_score)
    
    return key, decoded_text, best_score

def clean_repr(char_or_pair):
    rep = repr(char_or_pair)
    # Enlever les guillemets si c'est une chaîne de caractères
    if rep.startswith("'") and rep.endswith("'"):
        return rep[1:-1]
    return rep


def main():
    # Vérifier si les fichiers char_probabilities.txt et pair_probabilities.txt existent, sinon les créer
    if not os.path.exists("char_probabilities.txt") or not os.path.exists("pair_probabilities.txt"):
        url = "https://www.gutenberg.org/cache/epub/13846/pg13846.txt"
        text = load_text_from_web(url)
        url = "https://www.gutenberg.org/ebooks/4650.txt.utf-8"
        text = text + load_text_from_web(url)
        # Obtenir les probabilités de chaque caractère unique du texte
        char_count = Counter(text)
        total_chars = sum(char_count.values())
        char_probabilities = {char: count / total_chars for char, count in char_count.items()}
        # Trier les caractères en ordre décroissant de probabilité
        sorted_chars = sorted(char_probabilities, key=char_probabilities.get, reverse=True)

        # Écrire les probabilités dans un fichier
        with open("char_probabilities.txt", "w", encoding="utf-8") as f:
            for char in sorted_chars:
                f.write(f"{clean_repr(char)}: {char_probabilities[char]}\n")

        # Obtenir les probabilités de chaque paire de caractères du texte
        char_pairs = cut_string_into_pairs(text)
        pair_count = Counter(char_pairs)
        total_pairs = sum(pair_count.values())
        pair_probabilities = {pair: count / total_pairs for pair, count in pair_count.items()}
        # Trier les paires en ordre décroissant de probabilité
        sorted_pairs = sorted(pair_probabilities, key=pair_probabilities.get, reverse=True)

        # Écrire les probabilités dans un fichier
        with open("pair_probabilities.txt", "w", encoding="utf-8") as f:
            for pair in sorted_pairs:
                f.write(f"{clean_repr(pair)}: {pair_probabilities[pair]}\n")
    
    # Créer un dictionnaire des probabilités par caractère
    char_probabilities = {}
    with open("char_probabilities.txt", "r", encoding="utf-8") as f:
        for line in f:
            char, prob = line.rsplit(": ", 1)
            char_probabilities[char] = float(prob)
    
    # Créer un dictionnaire des probabilités par paire de caractères
    pair_probabilities = {}
    with open("pair_probabilities.txt", "r", encoding="utf-8") as f:
        for line in f:
            pair, prob = line.rsplit(": ", 1)
            pair_probabilities[pair] = float(prob)

    # On crée un dictionnaire de probabilités pondérées pour les caractères/paires de caractères
    weighted_probabilities = {}
    for char in char_probabilities:
        weighted_probabilities[char] = char_probabilities[char]*0.5
    for pair in pair_probabilities:
        weighted_probabilities[pair] = pair_probabilities[pair]*0.5

    # Stocker les mots de la langue française du fichier "liste.de.mots.francais.frgut.txt"
    with open("liste.de.mots.francais.frgut.txt", "r", encoding="utf-8") as f:
        french_words = {line.strip() for line in f}

    # Créer un set des mots de la langue française pour une recherche plus rapide
    french_words = set(french_words)

    # Chiffrer un texte (Traité sur la Tolérance de Voltaire)
    url = "https://www.gutenberg.org/cache/epub/42131/pg42131.txt"
    text = load_text_from_web(url)
    cle_secrete = gen_key(text_to_symbols(text))

    C = chiffrer2(text, cle_secrete)

    # Décoder le texte chiffré
    text_original = decode_text(C, weighted_probabilities, french_words)

    # Écrire le texte décodé dans un fichier
    with open("decoded_text.txt", "w") as f:
        f.write(text_original)

    return


if __name__ == '__main__':
    main()