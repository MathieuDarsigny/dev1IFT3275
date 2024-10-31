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

  rnd.seed(1337)
  int_keys = rnd.sample(list(range(l)),l)
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

def eval_score(text, french_words, pair_probabilities):
    # Scorer selon les probabilités des paires de caractères
    score = 0

    for i in range(0, len(text)):
        if i + 1 < len(text):
            pair = text[i][-1] + text[i + 1][0]
            score += pair_probabilities.get(pair, -1)

    return score

def decode_text(encoded_text, weighted_probabilities, french_words, pair_probabilities):
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
    # Au départ, on applique toutes les probabilités les plus proches 1 à 1 aux caractères les plus probables
    key = {}
    for i in range(len(sorted_bytes)):
        key[sorted_bytes[i]] = sorted_chars[i]

    decoded_text = []
    for byte in encoded_bytes:
        decoded_text.append(key[byte])
    print(f"Terminé le décodage initial")

    # Évaluer le message déchiffré avec le dictionnaire
    score = eval_score(decoded_text, french_words, pair_probabilities)
    print(f"Score initial: {score}")

    # Optimiser la clé
    key, decoded_text, score = optimize_key(key, weighted_probabilities, decoded_text, french_words, score, pair_probabilities)
    

    return ''.join(decoded_text)

def optimize_key(key, weighted_probabilities, decoded_text, french_words, score, pair_probabilities):
    # Répéter jusqu'à ce qu'il n'y ait plus d'améliorations significatives:
    #   - Pour chaque paire de caractères de la clé, si les fréquences des caractères ne sont pas trop éloignées, 
    #     échanger les caractères
    #   - Évaluer le message déchiffré avec le dictionnaire
    #   - Si le score est meilleur, mettre à jour la clé
    best_score = score
    recommencer = False
    for int1 in key:
        char1 = key[int1]
        if char1 is not None:
            for int2 in key:
                char2 = key[int2]
                if char2 is not None and char1 != char2:
                    # Vérifier si les fréquences des caractères ne sont pas trop éloignées
                    prob1 = weighted_probabilities[char1]
                    prob2 = weighted_probabilities[char2]
                    # Si la différence des probabilités est plus grande que la plus petite probabilité, ne pas échanger
                    #if abs(prob1 - prob2) > min(prob1, prob2):
                        #continue

                    # Évaluer le nouveau score
                    new_score = best_score
                    if char1 == "e " and char2 == "\\r\\n":
                        print(''.join(decoded_text))
                    for i in range(len(decoded_text)-1):
                        if decoded_text[i] == char1:
                            if decoded_text[i+1] == char1 or decoded_text[i+1] == char2:
                                old_pair = char1[-1] + decoded_text[i+1][0]
                                new_pair = char2[-1] + char2[0] if decoded_text[i+1] == char1 else char2[-1] + char1[0]
                                new_score -= pair_probabilities.get(old_pair, -1)
                                new_score += pair_probabilities.get(new_pair, -1)
                                if char1 == "e " and char2 == "\\r\\n":
                                    print(f"old_pair: {old_pair}, new_pair: {new_pair}, new_score: {new_score}")
                            else:
                                old_pair = char1[-1] + decoded_text[i+1][0]
                                new_pair = char2[-1] + decoded_text[i+1][0]
                                new_score -= pair_probabilities.get(old_pair, -1)
                                new_score += pair_probabilities.get(new_pair, -1)
                                if char1 == "e " and char2 == "\\r\\n":
                                    print(f"old_pair2: {old_pair}, new_pair2: {new_pair}, new_score: {new_score}")
                        elif decoded_text[i] == char2:
                            if decoded_text[i+1] == char1 or decoded_text[i+1] == char2:
                                old_pair = char2[-1] + decoded_text[i+1][0]
                                new_pair = char1[-1] + char1[0] if decoded_text[i+1] == char2 else char1[-1] + char2[0]
                                new_score -= pair_probabilities.get(old_pair, -1)
                                new_score += pair_probabilities.get(new_pair, -1)
                                if char1 == "e " and char2 == "\\r\\n":
                                    print(f"old_pair3: {old_pair}, new_pair3: {new_pair}, new_score: {new_score}")
                            else:
                                old_pair = char2[-1] + decoded_text[i+1][0]
                                new_pair = char1[-1] + decoded_text[i+1][0]
                                new_score -= pair_probabilities.get(old_pair, -1)
                                new_score += pair_probabilities.get(new_pair, -1)
                                if char1 == "e " and char2 == "\\r\\n":
                                    print(f"old_pair4: {old_pair}, new_pair4: {new_pair}, new_score: {new_score}")

                    # Si le score est meilleur
                    if new_score > best_score:
                        # Mettre à jour le score
                        best_score = new_score
                        score = new_score

                        # Mettre à jour le texte déchiffré
                        for i in range(len(decoded_text)):
                            if decoded_text[i] == char1:
                                decoded_text[i] = char2
                            elif decoded_text[i] == char2:
                                decoded_text[i] = char1
                        verif_score = eval_score(decoded_text, french_words, pair_probabilities)
                        if verif_score < best_score:
                            print(''.join(decoded_text))
                            print(f"ERREUR: Le score {verif_score} est plus bas que le meilleur score {best_score} après l'échange de {char1} et {char2}")
                            return key, decoded_text, best_score

                        # Échanger les caractères dans la clé
                        key[int1], key[int2] = char2, char1

                        # Recommencer la boucle externe puisqu'on a modifié la clé
                        recommencer = True
                        print(f"Échange de {char1} ({prob1}) et {char2} ({prob2})")
                        break
    if recommencer:
        if best_score>10000:
            return key, decoded_text, best_score
        print(f"\nNouveau score: {best_score}, recommencer l'optimisation")
        return optimize_key(key, weighted_probabilities, decoded_text, french_words, best_score, pair_probabilities)
    
    return key, decoded_text, best_score

def clean_repr(char_or_pair):
    rep = repr(char_or_pair)
    # Enlever les guillemets si c'est une chaîne de caractères
    if rep.startswith("'") and rep.endswith("'"):
        return rep[1:-1]
    return rep


def main():
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

    if not os.path.exists("probabilities.txt"):
        url = "https://www.gutenberg.org/cache/epub/13846/pg13846.txt"
        text = load_text_from_web(url)
        url = "https://www.gutenberg.org/ebooks/4650.txt.utf-8"
        text = text + load_text_from_web(url)

        cle_secrete = gen_key(text_to_symbols(text))
        C = chiffrer2(text, cle_secrete)
        # Sachant que le texte encodé est des 1 et des 0 où chaque tranche de 8 bits est un caractère

        # Déterminer le nombre de 0 à ajouter au début du texte pour obtenir un nombre de bits multiple de 8
        num_padding_bits = (8 - (len(C) % 8))%8

        # Ajouter les bits de padding au début du texte
        C = "0" * num_padding_bits + C

        # Séparer le texte encodé en tranches de 8 caractères
        encoded_bytes = [C[i:i + 8] for i in range(0, len(C), 8)]

        # Obtenir les probabilités de chaque séquence de 8 bits
        byte_probabilities = {byte: 0 for byte in set(encoded_bytes)}
        for byte in encoded_bytes:
            byte_probabilities[byte] += 1
        total_bytes = sum(byte_probabilities.values())
        byte_probabilities = {byte: count / total_bytes for byte, count in byte_probabilities.items()}

        char_to_probability = {}
        for char in cle_secrete:
            if cle_secrete[char] in byte_probabilities: 
                char_to_probability[char] = byte_probabilities[cle_secrete[char]]
            else:
                char_to_probability[char] = 0
        
        # Trier les caractères en ordre décroissant de probabilité
        sorted_chars = sorted(char_to_probability, key=char_to_probability.get, reverse=True)

        with open("probabilities.txt", "w", encoding="utf-8") as f:
            for char in sorted_chars:
                f.write(f"{clean_repr(char)}: {char_to_probability[char]}\n")
    
    '''# Créer un dictionnaire des probabilités par caractère
    char_probabilities = {}
    with open("probabilities.txt", "r", encoding="utf-8") as f:
        for line in f:
            char, prob = line.rsplit(": ", 1)
            char_probabilities[char] = float(prob)

    # Créer un dictionnaire des probabilités par paire de caractères
    pair_probabilities = {}
    with open("pair_probabilities.txt", "r", encoding="utf-8") as f:
        for line in f:
            pair, prob = line.rsplit(": ", 1)
            pair_probabilities[pair] = float(prob)

    # Stocker les mots de la langue française du fichier "liste.de.mots.francais.frgut.txt"
    with open("liste.de.mots.francais.frgut.txt", "r", encoding="utf-8") as f:
        french_words = {line.strip() for line in f}

    # Créer un set des mots de la langue française pour une recherche plus rapide
    french_words = set(french_words)

    # Chiffrer un texte (Traité sur la Tolérance de Voltaire)
    url = "https://www.gutenberg.org/cache/epub/42131/pg42131.txt"
    text = load_text_from_web(url)
    text = text[10000:11000]
    print(text)
    cle_secrete = gen_key(text_to_symbols(text))

    C = chiffrer2(text, cle_secrete)

    # Décoder le texte chiffré
    text_original = decode_text(C, char_probabilities, french_words, pair_probabilities)

    # Écrire le texte décodé dans un fichier
    with open("decoded_text.txt", "w") as f:
        f.write(text_original)
    '''
    return


if __name__ == '__main__':
    main()