import math
import os
import random as rnd
import re
import numpy as np
import requests
from collections import Counter
import time



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
    '''caracteres = list(set(list(text)))
    nb_caracteres = len(caracteres)
    nb_bicaracteres = 256-nb_caracteres
    bicaracteres = [item for item, _ in Counter(cut_string_into_pairs(text)).most_common(nb_bicaracteres)]'''
    symboles = ['b', 'j', '\r', 'J', '”', ')', 'Â', 'É', 'ê', '5', 't', '9', 'Y', '%', 'N', 'B', 'V', '\ufeff', 'Ê', '?', '’', 'i', ':', 's', 'C', 'â', 'ï', 'W', 'y', 'p', 'D', '—', '«', 'º', 'A', '3', 'n', '0', 'q', '4', 'e', 'T', 'È', '$', 'U', 'v', '»', 'l', 'P', 'X', 'Z', 'À', 'ç', 'u', '…', 'î', 'L', 'k', 'E', 'R', '2', '_', '8', 'é', 'O', 'Î', '‘', 'a', 'F', 'H', 'c', '[', '(', "'", 'è', 'I', '/', '!', ' ', '°', 'S', '•', '#', 'x', 'à', 'g', '*', 'Q', 'w', '1', 'û', '7', 'G', 'm', '™', 'K', 'z', '\n', 'o', 'ù', ',', 'r', ']', '.', 'M', 'Ç', '“', 'h', '-', 'f', 'ë', '6', ';', 'd', 'ô', 'e ', 's ', 't ', 'es', ' d', '\r\n', 'en', 'qu', ' l', 're', ' p', 'de', 'le', 'nt', 'on', ' c', ', ', ' e', 'ou', ' q', ' s', 'n ', 'ue', 'an', 'te', ' a', 'ai', 'se', 'it', 'me', 'is', 'oi', 'r ', 'er', ' m', 'ce', 'ne', 'et', 'in', 'ns', ' n', 'ur', 'i ', 'a ', 'eu', 'co', 'tr', 'la', 'ar', 'ie', 'ui', 'us', 'ut', 'il', ' t', 'pa', 'au', 'el', 'ti', 'st', 'un', 'em', 'ra', 'e,', 'so', 'or', 'l ', ' f', 'll', 'nd', ' j', 'si', 'ir', 'e\r', 'ss', 'u ', 'po', 'ro', 'ri', 'pr', 's,', 'ma', ' v', ' i', 'di', ' r', 'vo', 'pe', 'to', 'ch', '. ', 've', 'nc', 'om', ' o', 'je', 'no', 'rt', 'à ', 'lu', "'e", 'mo', 'ta', 'as', 'at', 'io', 's\r', 'sa', "u'", 'av', 'os', ' à', ' u', "l'", "'a", 'rs', 'pl', 'é ', '; ', 'ho', 'té', 'ét', 'fa', 'da', 'li', 'su', 't\r', 'ée', 'ré', 'dé', 'ec', 'nn', 'mm', "'i", 'ca', 'uv', '\n\r', 'id', ' b', 'ni', 'bl']
    return symboles

def decode(encoded_bytes, key):
    decoded_text = ""
    for i in range(0, len(encoded_bytes)):
        byte = encoded_bytes[i]
        char = key[byte]
        if char is None:
            char = ' '
        else:
            char = char.replace('&', '\r').replace('@', '\n').replace('<', '\ufeff')
        decoded_text += char
    return decoded_text

def clean_repr(char_or_pair):
    rep = repr(char_or_pair)
    # Enlever les guillemets si c'est une chaîne de caractères
    if rep.startswith("'") and rep.endswith("'"):
        return rep[1:-1]
    elif rep.startswith('"') and rep.endswith('"'):
        return rep[1:-1]
    return rep


def main():
    # Stocker les probabilités des 256 caractères possibles dans un fichier
    if not os.path.exists("probabilities.txt"):
        url = "https://www.gutenberg.org/cache/epub/13846/pg13846.txt"
        text = load_text_from_web(url)
        url = "https://www.gutenberg.org/ebooks/4650.txt.utf-8"
        text = text + load_text_from_web(url)

        cle_secrete = gen_key(text_to_symbols(text))
        print(cle_secrete)
        C = chiffrer2(text, cle_secrete)
        # Enlever tous les caractères non-binaires dans C
        C = re.sub(r"[^01]", "", C)

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
                # Remplacer les caractères spéciaux par d'autres représentations qui ne sont pas des symboles de la clé pour éviter des maux de tête
                clean_char = clean_repr(char.replace('\r', '&').replace('\n', '@').replace('\ufeff', '<'))
                f.write(f"{clean_char}: {char_to_probability[char]}\n")
    
    if not os.path.exists("word_probabilities.txt"):
        url = "https://www.gutenberg.org/cache/epub/42131/pg42131.txt"
        text = load_text_from_web(url)
        url = "https://www.gutenberg.org/cache/epub/68355/pg68355.txt"
        text = text + load_text_from_web(url)
        text = text[10000:] # Enlever les 10 000 premiers caractères
        # Enlever les premiers caractères jusqu'au premier mot complet
        text = text[text.index(' ') + 1:]

        # Séparer le texte en mots
        words = text.split()

        # Compter le nombre d'occurrences de chaque mot
        word_counts = Counter(words)

        # Calculer les probabilités de chaque mot
        total_words = sum(word_counts.values())
        word_probabilities = {word: count / total_words for word, count in word_counts.items()}
        sorted_words = sorted(word_probabilities, key=word_probabilities.get, reverse=True)

        with open("word_probabilities.txt", "w", encoding="utf-8") as f:
            for word in sorted_words:
                f.write(f"{word}: {word_probabilities[word]}\n")

    # Chiffrer le texte du test.py fourni:
    # Charger le premier corpus et enlever les 10 000 premiers caractères
    url1 = "https://www.gutenberg.org/ebooks/13846.txt.utf-8"
    corpus1 = load_text_from_web(url1)

    # Charger le deuxième corpus et enlever les 10 000 premiers caractères
    url2 = "https://www.gutenberg.org/ebooks/4650.txt.utf-8"
    corpus2 = load_text_from_web(url2)

    # Combiner les deux corpus
    corpus = corpus1 + corpus2
    rnd.seed(time.time())

    a = rnd.randint(3400, 7200)
    b = rnd.randint(96000, 125000)
    l = a+b
    c = rnd.randint(0, len(corpus)-l)

    M = corpus[c:c+l]
    print("Longueur du message à décoder: ", len(M))

    cle_secrete = gen_key(text_to_symbols(M))

    C = chiffrer2(M, cle_secrete)

    # Décoder le texte chiffré
    decrypt(C, cle_secrete)

    return

def decrypt(encoded_text, cle_secrete):
    # ---------------------- Préparation des données ----------------------

    # On utilise la clé secrète à des fins de tests
    secret_key_byte_to_char = {}
    for char, byte in cle_secrete.items():
        secret_key_byte_to_char[int(byte,2)] = char

    # Stocker les mots de la langue française du fichier "liste.de.mots.francais.frgut.txt"
    with open("liste.de.mots.francais.frgut.txt", "r", encoding="utf-8") as f:
        french_words = {line.strip() for line in f}

    # Stocker les probabilités des mots de la langue française du fichier "word_probabilities.txt"
    word_probabilities = {}
    with open("word_probabilities.txt", "r", encoding="utf-8") as f:
        for line in f:
            word, prob = line.rsplit(": ", 1)
            word_probabilities[word] = float(prob)

    # Créer un dictionnaire des probabilités par caractère basé sur une clé secrète générée sur un long texte avec la méthode donnée
    char_probabilities = {}
    with open("probabilities.txt", "r", encoding="utf-8") as f:
        for line in f:
            char, prob = line.rsplit(": ", 1)
            char_probabilities[char] = float(prob)

    # Trier char_probabilities en ordre décroissant de probabilité
    sorted_chars = sorted(char_probabilities, key=char_probabilities.get, reverse=True)

    # Déterminer le nombre de 0 à ajouter au début du texte pour obtenir un nombre de bits multiple de 8
    num_padding_bits = (8 - (len(encoded_text) % 8))%8

    # Ajouter les bits de padding au début du texte
    encoded_text = "0" * num_padding_bits + encoded_text

    # Séparer le texte encodé en bytes (convertis en entiers)
    encoded_bytes = [int(encoded_text[i:i + 8],2) for i in range(0, len(encoded_text), 8)]

    # Obtenir les probabilités de chaque byte
    byte_probabilities = {byte: 0 for byte in set(encoded_bytes)}
    for byte in encoded_bytes:
        byte_probabilities[byte] += 1
    total_bytes = sum(byte_probabilities.values())
    byte_probabilities = {byte: count / total_bytes for byte, count in byte_probabilities.items()}

    # Trier les séquences de 8 bits en ordre décroissant de probabilité
    sorted_bytes = sorted(byte_probabilities, key=byte_probabilities.get, reverse=True)

    # ------------------------------ DÉCODAGE--------------------------------------------------
    key = [None] * 256
    # ______________ 1. Hardcode la substitution 'e ' pour le byte le plus probable ___________
    key[sorted_bytes[0]] = 'e '
    print("Étape 1: Substitution de '" + str(sorted_bytes[0]) + "' par 'e '")
    print("Validation avec la clé secrète: '" + secret_key_byte_to_char[sorted_bytes[0]] + "'")

    # Print les 10 autres bytes les plus probables ainsi que la clé secrète pour déboggage
    print("10 autres bytes les plus probables: clé-dictionnaire vs clé secrète")
    for i in range(1,11):
        print("Char : '" + sorted_chars[i] + "' =  char '" + secret_key_byte_to_char[sorted_bytes[i]] +"'")

    # ______________________________ 2. Déchiffrer le caractère ' ' ___________________________
    
    '''
    'e ' = le byte le plus fréquent.
    On peut ensuite directement deviner le byte associé à ' '. En effet, 'e ' ne sera jamais suivi de ' '.
    On prend tous les bytes du texte qui ne suivent jamais le caractère 'e '. 
    Le byte avec la probabilité la plus proche de notre clé-dictionnaire est l'espace ' '.
    Ex.:  Probabilité de la clé-dictionnaire: ' ':0.012297671705633598
    '''
    def relative_difference(real_value, observed_value, epsilon=1e-8):
        """
        Calcul de l'écart relatif entre:
            - value1: Valeur théorique
            - value2: Valeur observée
        """
        return abs(real_value - observed_value) / (real_value + epsilon)
    
    # Les bytes qui suivent "e " dans le texte
    follows_e_space = []
    prev_byte = None
    for byte in encoded_bytes:
        if key[byte] is None and prev_byte is not None and key[prev_byte] == 'e ':
            if byte not in follows_e_space:
                follows_e_space.append(byte)
        prev_byte = byte

    # Obtenir les probabilités des bytes qui ne suivent jamais 'e '
    dont_follow_e_space = []
    for byte in byte_probabilities:
        if not byte in follows_e_space:
            dont_follow_e_space.append(byte)
    
    # Trouver le byte dont la probabilité est la plus proche de celle de ' '
    closest_byte_to_space = None
    closest_diff = float('inf')
    for byte in dont_follow_e_space:
        diff = relative_difference(real_value=char_probabilities[' '], observed_value=byte_probabilities[byte])
        if diff < closest_diff:
            closest_diff = diff
            closest_byte_to_space = byte
    
    # Ajouter la substitution de ' ' à la clé
    key[closest_byte_to_space] = ' '
    print("Étape 2: Substitution de '" + str(closest_byte_to_space) + "' par ' '")
    print("Validation avec clé secrète: '" + secret_key_byte_to_char[closest_byte_to_space] + "'")

    # ________________ 3. Déchiffrer tous les bytes de forme 'e ' + BYTE + 'e ' _________________
    '''
    Maintenant qu'on connaît où sont encodés les espaces simples, on sait que s'il y a (' '|'e ') + BYTE + 'e ', c'est un mot de maximum 3 lettres qui se termine par e.
    
        a) On va chercher dans la langue française tous les mots de 2 ou 3 lettres qui se terminent par e, et on enlève le 'e'. 
            i.  On associe chaque groupe de 1 à 2 lettres formé à sa probabilité du clé-dictionnaire. 
            ii. On prend aussi les probabilités d'apparition du mot dans la langue française.
        b) On va chercher tous les bytes qui sont de forme (' '|'e ') + BYTE + 'e ' et leur fréquence.
        
        c) On trouve chacune des substitutions pour les bytes à l'aide d'une comparaison entre fréquence du groupe et fréquence du byte.
            i. On supplémente aussi l'analyse des fréquences avec les probabilités d'apparition du mot dans la langue française pour s'assurer à 100% de faire les bonnes substitutions.

    '''
    # a) Obtenir les mots de 2 ou 3 lettres qui se terminent par 'e' dans la langue française
    french_words_2_3 = {word[:-1]: 0 for word in french_words if (len(word) == 3 or len(word) == 2) and word[-1] == 'e'}

    # i. Associer chaque groupe de 1 à 2 lettres formé à sa probabilité du clé-dictionnaire
    char_to_probability = {}
    for word in french_words_2_3:
        word_without_e = char[0:-1]
        if word_without_e in char_probabilities:
            char_to_probability[word_without_e] = char_probabilities[word_without_e]
        else: 
            char_to_probability[word_without_e] = 0

    # ii. Associer chaque mot de 2 à 3 lettres à sa probabilité d'apparition dans la langue française
    #       - L'information se trouve déjà dans word_probabilities

    def capture_groups(start_str, n_bytes_min, n_bytes_max, end_str, decrypt_when_possible=False, bytes_with_starting_spaces=[False]*256, bytes_with_ending_spaces=[False]*256):
        '''
        Arguments
            - start_str: Une regex qui délimite le début des mots à capturer (ex.: '[^ ]? ' pour indiquer que la chaîne commence par n'importe quel caractère suivi d'un espace)
                start_str doit toujours comporter un espace
            - n_bytes_min: longueur minimale de la chaîne du milieu en bytes
            - n_bytes_max: longueur maximale de la chaîne du milieu en bytes
            - end_str: Une regex qui délimite la fin des mots à capturer (ex.: ' [^ ]?' pour indiquer que la chaîne se termine par un espace suivi de n'importe quel caractère)
                end_str doit toujours comporter un espace
            - decrypt_when_possible: Si vrai, les bytes déjà substitués sont transformés en leur représentation en string dans les groupes. Faux par défaut
            - bytes_with_starting_spaces: Optionnel. Liste de Booléens qui indique si le byte i commence par un espace.
                Si le byte i commence par un espace, on segmente le groupe en conséquence
                Note: ' ' ne doit pas être True
            - bytes_with_ending_spaces: Optionnel. Liste de Booléens qui indique si le byte i se termine par un espace.
                Si le byte i se termine par un espace, on segmente le groupe en conséquence
                Note: ' ' ne doit pas être True
        
        Retourne
            - Une liste des mots du texte qui correspondent aux bytes à l'intérieur de start_str et end_str
        '''
        if not (start_str[0] == ' ' or start_str[-1] == " " or end_str == ' ' or end_str[-1] == ' '):
            raise ValueError("start_str et end_str doivent contenir au moins un espace pour séparer les mots")

        # Note: On utilise le symbole '~' pour indiquer un caractère inconnu/indécrypté
        groups = []
        group = []
        capturing = False
        for i in range(len(encoded_bytes)-1):
            byte = encoded_bytes[i]
            next_byte = encoded_bytes[i+1]
            char = key[byte]

            # Si char est None, vérifier bytes_with_starting_spaces ou bytes_with_ending_spaces et ajuster char
            if char is None:
                if bytes_with_starting_spaces[byte]:
                    char = ' ~'
                elif bytes_with_ending_spaces[byte]:
                    char = '~ '
            char2 = key[next_byte]
            # Si char2 est None, vérifier bytes_with_starting_spaces ou bytes_with_ending_spaces et ajuster char2
            if char2 is None:
                if bytes_with_starting_spaces[next_byte]:
                    char2 = ' ~'
                elif bytes_with_ending_spaces[next_byte]:
                    char2 = '~ '
            
            # Print i, len(group), byte, next_byte, char, char2 pour déboggage
            #print("i:", i, "len(group):", len(group),"byte:", byte,"next_byte:", next_byte,"char1", char,"char2", char2)


            if char is not None:
                # On capture présentement le mot
                if capturing:
                    # On trouve la fin du mot sur 1 byte
                    if re.fullmatch(string=char, pattern=end_str):
                        capturing = False
                        if len(group) >= n_bytes_min and len(group) <= n_bytes_max:
                            groups.append(group)
                        group = []
                    # On trouve la fin du mot sur 2 bytes
                    elif char2 is not None and re.fullmatch(string=char + char2, pattern=end_str):
                        capturing = False
                        if len(group) >= n_bytes_min and len(group) <= n_bytes_max:
                            groups.append(group)
                        group = []
                    # On n'a pas matché de fin de mot mais nous avons un espace. On arrête de capturer le mot
                    elif char2 is not None and ' ' in char2:
                        capturing = False
                        group = []
                    # On continue de capturer le mot
                    elif decrypt_when_possible:
                        group.append(char)
                    # On laisse un caractère bidon si on ne veut pas décrypter
                    else:
                        group.append(byte)
                # On ne capture pas présentement le mot
                else:
                    # On commence à capturer le mot sur 1 byte
                    if re.fullmatch(string=char, pattern=start_str):
                        capturing = True
                        group = []
                    # On commence à capturer le mot sur 2 bytes
                    elif char2 is not None and re.fullmatch(string=char + char2, pattern=start_str):
                        capturing = True
                        i += 1
                        group = []
            # char est None
            else:
                if char2 is not None:
                    # On capture présentement le mot
                    if capturing:
                        group.append(byte)
                        # On trouve la fin du mot sur le prochain byte
                        if re.fullmatch(string=char2, pattern=end_str):
                            capturing = False
                            if len(group) >= n_bytes_min and len(group) <= n_bytes_max:
                                groups.append(group)
                            group = []
                        # On n'a pas matché de fin de mot mais nous avons un espace. On arrête de capturer le mot
                        elif ' ' in char2:
                            capturing = False
                            group = []
                    # On ne capture pas présentement le mot
                    else:
                        # On commence à capturer le mot sur le prochain byte
                        if re.fullmatch(string=char2, pattern=start_str):
                            capturing = True
                            group = []
                # char1 et char2 est None
                else:
                    if capturing:
                        if len(group) <= n_bytes_max:
                            group.append(byte)
                        else:
                            capturing = False
                            group = []
        return groups

    """ # Test de la fonction capture_groups OK
    # B1 = 'e ', B2 = ' '
    # C = B1 B3 B2 B3 B1 B4 B2 B6 B7 B1 B8 B1
    # M = e ,~, ,e ,~, ,~,~,e ,~,e 
    # On s'attend aux groupes [B3, B8]
    encoded_bytes = [1, 3, 2, 3, 1, 4, 2, 6, 7, 1, 8, 1]
    key = [None] * 256
    key[1] = 'e '
    key[2] = ' ' 
    groups = capture_groups(start_str=r'[^ ]? ', n_bytes_min=1, n_bytes_max=2, end_str=r'e ')
    """

    # b) Obtenir les bytes qui sont de forme (' '|'e ') + BYTE + 'e ' et leur fréquence
    groups = capture_groups(start_str=r'[^ ]? ', n_bytes_min=1, n_bytes_max=1, end_str=r'e ')

    print("Nombre de groupes capturés: ", len(groups))
    for group in groups:
        for byte in group:
            print("Byte : '" + str(byte) + "' =  char '" + secret_key_byte_to_char[byte] +"'")
            
    # c) On trouve chacune des substitutions pour les bytes à l'aide d'une comparaison entre fréquence du groupe et fréquence du byte.
    #        i. On supplémente aussi l'analyse des fréquences avec les probabilités d'apparition du mot dans la langue française pour s'assurer à 100% de faire les bonnes substitutions.
    
    # Calculer les fréquences de chaque groupe dans groups

    # Bug: On ne peut pas utiliser Counter(groups) car les listes ne sont pas hashables
    group_counts = Counter(groups)
    total_groups = sum(group_counts.values())
    group_probabilities = {group: count / total_groups for group, count in group_counts.items()}


    # Obtenir un sous-ensemble de word_probabilities qui contient uniquement les mots de french_words_2_3
    word_probabilities_2_3 = {word: prob for word, prob in word_probabilities.items() if word in french_words_2_3}

    # Normaliser les probabilités des mots de 2 à 3 lettres
    total_words_2_3 = sum(word_probabilities_2_3.values())
    word_probabilities_2_3 = {word: prob / total_words_2_3 for word, prob in word_probabilities_2_3.items()}
    
    # Pour chaque byte de groups, trouver les 1 à 2 caractères qui correspond le mieux en comparant 2 métriques à la fois:
    for byte, group_prob in group_probabilities:
        best_char = None
        best_diff = float('inf')
        # Chaque char est un mot de 2 à 3 lettres qui se termine par 'e'
        for char in char_to_probability:
            # Fréquence du byte dans le texte encodé vs fréquence des caractères dans le clé-dictionnaire
            diff = relative_difference(real_value=char_to_probability[char], observed_value=byte_probabilities[byte])
            # Fréquence du groupe vs fréquence normalisée du mot de 2 à 3 lettres dans la langue française
            diff2 = relative_difference(real_value=word_probabilities_2_3[char+"e"], observed_value=group_prob)

            # Moyenne des deux différences
            avg_diff = (diff + diff2)/2
            if avg_diff < best_diff:
                best_diff = avg_diff
                best_char = char
        
        # Ajouter la substitution à la clé
        key[byte] = best_char
        print("Étape 3: Substitution de '" + byte + "' par '" + best_char + "' pour former le mot de 2 à 3 lettres '" + best_char + "e'")
        


    # Écrire le fichier décodé
    decoded_text = decode(encoded_bytes, key)
    with open("decoded_text.txt", "w", encoding="utf-8") as f:
        f.write(decoded_text)

    return






if __name__ == '__main__':
    main()
