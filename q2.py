import os
import random as rnd
import re
import requests
from collections import Counter
import time

global allow_rand
allow_rand = True

def sequence_fr_text(*urls):
    """Télécharge et traite les textes des URLs données."""
    corpus = ""
    
    # Charger et traiter chaque URL
    for url in urls:
        text = load_text_from_web(url)
        
        # Enlever les 5000 premiers et derniers caractères
        if len(text) > 10000:
            text = text[5000:-5000]
        
        # Enlever les premiers caractères jusqu'au premier mot complet, puis les derniers caractères jusqu'au dernier 
        # mot complet, en gardant cette fois-ci l'espace
        text = text[text.index(' ') + 1:text.rindex(' ') + 1]

        # Ajouter le texte traité au corpus
        corpus += text
    
    # Chiffrer le corpus combiné
    symbols = text_to_symbols(corpus)
    key = gen_key(symbols)
    sequenced_text = M_vers_symboles(corpus, key)
    
    return sequenced_text

def frequence_matrixes(dimension, fr_text, encoded_bytes):
        """
        Crée deux matrices de fréquences (bytes et langue française) de dimension donnée.
        Pour dimension = 1: les matrices sont des dictionnaires simples.
            Ex.: Pour obtenir la liste des bytes qui suivent le byte 1 on fait matrice[1]
        Pour dimension >1, les matrices sont en réalité des dictionnaires de tuples.
            Ex.: Pour obtenir la liste des bytes qui suivent le byte 1 et 3 on fait matrice_bytes[(1,3)]

        Arguments:
            dimension: La dimension de la matrice voulue
            texte_fr: Un tableau d'un texte en français dont les caractères sont séquencés selon l'encodage en bytes
            encoded_bytes: Les bytes encodés
        """
        matrice_bytes = {}
        matrice_fr = {}
        for texte, matrice in ((encoded_bytes, matrice_bytes), (fr_text, matrice_fr)):
            for i in range(len(texte) - dimension + 1):
                window = texte[i:i + dimension]
                if dimension == 1:
                    char_or_byte = window[0]
                    if char_or_byte in matrice:
                        matrice[char_or_byte] += 1
                    else:
                        matrice[char_or_byte] = 1
                else:
                    chars_or_bytes = tuple(window)
                    if chars_or_bytes in matrice:
                        matrice[chars_or_bytes] += 1
                    else:
                        matrice[chars_or_bytes] = 1
        return matrice_bytes, matrice_fr


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
    global allow_rand
    a, b, l, c = 0, 0, 0, 0
    M = ""
    if allow_rand:
        a = rnd.randint(3400, 7200)
        b = rnd.randint(96000, 125000)
        l = a+b
        c = rnd.randint(0, len(corpus)-l)
        M = corpus[c:c+l]
    else:
        M = corpus[5000:-5000]
    

    
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
        secret_key_byte_to_char[int(byte,2)] = char.replace('\r', '&').replace('\n', '@').replace('\ufeff', '<')

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

    # Écrire le message chiffré dans un fichier, où tous les bytes sont décodés et séparés par | pour faciliter la lecture
    with open("encoded_text_full_except_5000.txt", "w", encoding="utf-8") as f:
        for byte in encoded_bytes:
            f.write(secret_key_byte_to_char[byte] + "|")

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
    """ print("10 autres bytes les plus probables: clé-dictionnaire vs clé secrète")
    for i in range(1,11):
        print("Char : '" + sorted_chars[i] + "' =  char '" + secret_key_byte_to_char[sorted_bytes[i]] +"'") """

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
    
    # ------------------------- TOUS LES BYTES QUI SUIVENT 'e ' -------------------------

    # Les bytes qui suivent "e " dans le texte. Ces bytes sont garantis de ne pas commencer par un espace.
    no_starting_space = []
    prev_byte = None
    for byte in encoded_bytes:
        if key[byte] is None and prev_byte is not None and key[prev_byte] == 'e ':
            if byte not in no_starting_space:
                no_starting_space.append(byte)
        prev_byte = byte
    
    #print("\nVoici les bytes qui suivent 'e ' dans le texte ainsi que leurs fréquences observées vs théoriques:")
    no_starting_space_pairs = []
    for byte in no_starting_space:
        no_starting_space_pairs.append((byte, byte_probabilities[byte]))

    # Trier les pairs en ordre décroissant de probabilité
    no_starting_space_pairs = sorted(no_starting_space_pairs, key=lambda x: x[1], reverse=True)
    i = 0
    for (byte, prob) in no_starting_space_pairs:
        i+=1
        #print(str(i) + ": '" + secret_key_byte_to_char[byte] + "' Obs: " + str(prob) + " | Théorique: " + str(char_probabilities[secret_key_byte_to_char[byte]]))

    # Obtenir les probabilités des bytes qui commencent peut-être par " " en inversant no_starting_space
    maybe_starting_space = []
    for byte in byte_probabilities:
        if not byte in no_starting_space:
            maybe_starting_space.append(byte)
    
    #print("\nVoici les bytes qui ne suivent jamais 'e ' dans le texte ainsi que leurs fréquences observées vs théoriques:")
    maybe_starting_space_pairs = []
    for byte in maybe_starting_space:
        maybe_starting_space_pairs.append((byte, byte_probabilities[byte]))

    # Trier les bytes en ordre décroissant de probabilité
    maybe_starting_space_pairs = sorted(maybe_starting_space_pairs, key=lambda x: x[1], reverse=True)
    i = 0
    for (byte, prob) in maybe_starting_space_pairs:
        i+=1
        #print(str(i) + ": '" + secret_key_byte_to_char[byte] + "' Obs: " + str(prob) + " | Théorique: " + str(char_probabilities[secret_key_byte_to_char[byte]]))
    

    # ------------------------- TOUS LES BYTES QUI PRÉCÈDENT 'e ' -------------------------
    # Les bytes qui précèdent "e " dans le texte. Ces bytes sont garantis de ne pas se terminer par un espace, car " e " n'est pas un mot.
    no_ending_space = []
    prev_byte = None
    for byte in encoded_bytes:
        if key[byte] is not None and key[byte] == 'e ' and prev_byte is not None:
            if prev_byte not in no_ending_space:
                no_ending_space.append(prev_byte)
        prev_byte = byte
    
    #print("\nVoici les bytes qui précèdent 'e ' dans le texte ainsi que leurs fréquences observées vs théoriques:")
    no_ending_space_pairs = []
    for byte in no_ending_space:
        no_ending_space_pairs.append((byte, byte_probabilities[byte]))

    # Trier les pairs en ordre décroissant de probabilité
    no_ending_space_pairs = sorted(no_ending_space_pairs, key=lambda x: x[1], reverse=True)
    i = 0
    for (byte, prob) in no_ending_space_pairs:
        i+=1
        #print(str(i) + ": '" + secret_key_byte_to_char[byte] + "' Obs: " + str(prob) + " | Théorique: " + str(char_probabilities[secret_key_byte_to_char[byte]]))

    # Obtenir les probabilités des bytes qui ne précèdent jamais 'e '
    maybe_ending_space = []
    for byte in byte_probabilities:
        if not byte in no_ending_space:
            maybe_ending_space.append(byte)
    
    #print("\nVoici les bytes qui ne précèdent jamais 'e ' dans le texte ainsi que leurs fréquences observées vs théoriques:")
    maybe_ending_space_pairs = []
    for byte in maybe_ending_space:
        maybe_ending_space_pairs.append((byte, byte_probabilities[byte]))

    # Trier les pairs en ordre décroissant de probabilité
    maybe_ending_space_pairs = sorted(maybe_ending_space_pairs, key=lambda x: x[1], reverse=True)
    i = 0
    for (byte, prob) in maybe_ending_space_pairs:
        i+=1
        #print(str(i) + ": '" + secret_key_byte_to_char[byte] + "' Obs: " + str(prob) + " | Théorique: " + str(char_probabilities[secret_key_byte_to_char[byte]]))

    # On a:
    #   Une liste de bytes qui ne commencent pas par des espaces
    #   Une liste de bytes qui ne se terminent pas par des espaces
    #   Notre objectif final est de connaître tous les bytes qui ne contiennent pas d'espaces

    # On crée un set de bytes pour lesquels on est certain qu'ils ne contiennent pas d'espaces
    e1_set = set(no_starting_space)
    e2_set = set(no_ending_space)
    
    e2_sauf_e1 = (e2_set - e1_set)

    bytes_no_spaces = (e1_set & e2_set)
    """ print("Nombre de caractères sans espaces:" + str(len(bytes_no_spaces)))
    # Imprimer tous les bytes qui ne contiennent pas d'espaces
    print("\nVoici les bytes qui ne contiennent pas d'espaces:")
    for byte in bytes_no_spaces:
        print("'" + secret_key_byte_to_char[byte] + "'") """

    # Trouver les bytes qui contiennent potentiellement des espaces
    """ print("\nVoici les bytes qui contiennent potentiellement des espaces:")
    maybe_space_bytes = []
    i = 0
    for byte in sorted_bytes:
        if byte not in bytes_no_spaces:
            maybe_space_bytes.append(byte)
            i+=1
            print(str(i) + ": '" + secret_key_byte_to_char[byte] + "' Obs: " + str(byte_probabilities[byte]) + " | Théorique: " + str(char_probabilities[secret_key_byte_to_char[byte]])) """

    # ________________ 3. Capturer les groupes 'e ' + BYTE + 'e ' _________________

    def capture_groups(start_str, n_bytes_min, n_bytes_max, end_str, decrypt_when_possible=False, bytes_with_starting_spaces=[False]*256, bytes_with_ending_spaces=[False]*256, include_start_and_end=False):
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

        # Note: On utilise le symbole '~' pour indiquer un caractère inconnu
        groups = []
        group = []
        capturing = False
        start_str_matched = ""
        for i in range(2, len(encoded_bytes)-2):
            debug = False
            byte = encoded_bytes[i]
            next_byte = encoded_bytes[i+1]
            char = key[byte]
            if not capturing:
                start_str_matched = ""

            """ if (char is None and secret_key_byte_to_char[byte] != 'e ' and secret_key_byte_to_char[byte] != 'qu') and secret_key_byte_to_char[next_byte] == 'e ' and secret_key_byte_to_char[encoded_bytes[i+2]] == 'qu':
                debug = True
                print("\nCas now='None' + next='e ' + next2='qu'")
            elif secret_key_byte_to_char[byte] == 'e ' and secret_key_byte_to_char[next_byte] == 'qu' and secret_key_byte_to_char[encoded_bytes[i+2]] == 'e ':
                debug = True
                print("\nCas now='e ' + next = 'qu' + 'e '")
            elif secret_key_byte_to_char[encoded_bytes[i-1]] == 'e ' and secret_key_byte_to_char[byte] == 'qu' and secret_key_byte_to_char[encoded_bytes[i+1]] == 'e ':
                debug = True
                print("\nCas prev='e ', now='qu', next='e '")
            elif secret_key_byte_to_char[encoded_bytes[i-2]] == 'e ' and secret_key_byte_to_char[encoded_bytes[i-1]] == 'qu' and secret_key_byte_to_char[byte] == 'e ':
                debug = True
                print("\nCas prev='e ' + 'qu', now='e '") """

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

            if char is not None:
                # On capture présentement le mot
                if capturing:
                    if debug: print("char1 != None and capturing")
                    # On trouve la fin du mot sur 1 byte
                    if re.fullmatch(string=char, pattern=end_str):
                        if debug: print("char == end_str")
                        capturing = False
                        if len(group) >= n_bytes_min and len(group) <= n_bytes_max:
                            if include_start_and_end:
                                group.append(char)
                                group = [start_str_matched] + group
                            groups.append(group)
                        group = []
                    # On trouve la fin du mot sur 2 bytes
                    elif char2 is not None and re.fullmatch(string=char + char2, pattern=end_str):
                        if debug: print("char + char2 == end_str")
                        capturing = False
                        if len(group) >= n_bytes_min and len(group) <= n_bytes_max:
                            if include_start_and_end:
                                group.append(char + char2)
                                group = [start_str_matched] + group
                            groups.append(group)
                        group = []
                    # On n'a pas matché de fin de mot mais on doit s'arrêter (mot trop long ou espace)
                    elif len(group) > n_bytes_max or (char2 is not None and ' ' in char2):
                        if debug: print("len(group) > n_bytes_max or ' ' in char2")
                        capturing = False
                        group = []
                    # On continue de capturer le mot
                    elif decrypt_when_possible:
                        if debug: print("decrypt_when_possible")
                        group.append(char)
                    # On laisse un caractère bidon si on ne veut pas décrypter
                    else:
                        if debug: print("not decrypt_when_possible")
                        group.append(byte)
                # On ne capture pas présentement le mot
                else:
                    if debug: print("char1 != None and not capturing")
                    # On commence à capturer le mot sur 1 byte
                    if re.fullmatch(string=char, pattern=start_str):
                        if debug: print("char == start_str")
                        capturing = True
                        start_str_matched = char
                        group = []
                    # On commence à capturer le mot sur 2 bytes
                    elif char2 is not None and re.fullmatch(string=char + char2, pattern=start_str):
                        if debug: print("char + char2 == start_str")
                        capturing = True
                        start_str_matched = char + char2
                        i += 1
                        group = []
            # char est None
            else:
                if char2 is not None:
                    # On capture présentement le mot
                    if capturing:
                        if debug: print("char2 != None and capturing")
                        group.append(byte)
                        # On trouve la fin du mot sur le prochain byte
                        if re.fullmatch(string=char2, pattern=end_str):
                            if debug: print("char2 == end_str")
                            capturing = False
                            if len(group) >= n_bytes_min and len(group) <= n_bytes_max:
                                if include_start_and_end:
                                    group.append(char2)
                                    group = [start_str_matched] + group
                                groups.append(group)
                            group = []
                        # On n'a pas matché de fin de mot mais on doit s'arrêter (mot trop long ou espace)
                        elif len(group) > n_bytes_max or ' ' in char2:
                            if debug: print("len(group) > n_bytes_max or ' ' in char2")
                            capturing = False
                            group = []
                # char1 et char2 est None
                else:
                    if capturing:
                        if debug: print("chars == None and capturing")
                        if len(group) <= n_bytes_max:
                            group.append(byte)
                        else:
                            capturing = False
                            group = []
                    else:
                        if debug: print("chars == None and not capturing")
            # Print i, len(group), byte, next_byte, char, char2 pour déboggage
            if debug: print("i:", i, "group:", (group),"byte:", byte,"next_byte:", next_byte,"char1", char,"char2", char2)
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
    groups = capture_groups(start_str=r'e ', n_bytes_min=1, n_bytes_max=1, end_str=r'e ')

    """ print("Nombre de groupes capturés: ", len(groups))
    for group in groups:
        for byte in group:
            print("Byte : '" + str(byte) + "' =  char '" + secret_key_byte_to_char[byte] +"'") """
    
    # Trouver le nombre d'occurences de chaque byte dans les groupes
    occurences_group = {}
    for group in groups:
        for byte in group:
            if byte in occurences_group:
                occurences_group[byte] += 1
            else:
                occurences_group[byte] = 1
    
    # Trier les bytes en ordre décroissant d'occurences
    sorted_occurences_group = sorted(occurences_group, key=occurences_group.get, reverse=True)
    
    # Le premier byte est "qu". On ignore le reste qui est trop rare
    # On peut donc les substituer directement
    key[sorted_occurences_group[0]] = 'qu'
    # Vérifier la substitution avec la clé secrète
    print("Étape 3: Substitution de '" + str(sorted_occurences_group[0]) + "' par 'qu'")
    print("Validation avec la clé secrète: '" + secret_key_byte_to_char[sorted_occurences_group[0]] + "'")

    # On cherche maintenant tous les bytes qui suivent "e " + "qu" sauf ceux déjà décodés
    bytes_after_e_qu = {}
    for i in range(2, len(encoded_bytes)):
        byte = encoded_bytes[i]
        prev_prev_char = key[encoded_bytes[i-2]]
        prev_char = key[encoded_bytes[i-1]]
        if prev_prev_char == 'e ' and prev_char == 'qu' and key[byte] == None:
            if byte in bytes_after_e_qu:
                bytes_after_e_qu[byte] += 1
            else:
                bytes_after_e_qu[byte] = 1
    
    # Trier les bytes en ordre décroissant d'occurences
    sorted_bytes_after_e_qu = sorted(bytes_after_e_qu, key=bytes_after_e_qu.get, reverse=True)

    # Imprimer les premiers bytes après "e " + "qu" avec leur probabilité observée vs réelle
    #print("\nBytes après 'e ' + 'qu':")
    i = 0
    for byte in sorted_bytes_after_e_qu:
        i+=1
        #print(str(i) + ": Freq.: " + str(bytes_after_e_qu[byte]) + " | '" + secret_key_byte_to_char[byte] + "' Obs: " + str(byte_probabilities[byte]) + " | Théorique: " + str(char_probabilities[secret_key_byte_to_char[byte]]))

    # Le plus fréquent est le "i ".
    # On peut donc le substituer directement
    key[sorted_bytes_after_e_qu[0]] = 'i '
    # Vérifier la substitution avec la clé secrète
    print("Étape 3: Substitution de '" + str(sorted_bytes_after_e_qu[0]) + "' par 'i '")
    print("Validation avec la clé secrète: '" + secret_key_byte_to_char[sorted_bytes_after_e_qu[0]] + "'")

    groups = capture_groups(start_str=r'(e |i )', n_bytes_min=1, n_bytes_max=1, end_str=r'(e |i )', include_start_and_end=True)
    #print("Nombre de groupes capturés: ", len(groups))
    for group in groups:
        group_str = ""
        for byte in group:
            # Vérifier si byte est une String
            if isinstance(byte, str):
                group_str += byte
            else:
                group_str += secret_key_byte_to_char[byte]
        #print("Mot '" + group_str + "'")

    '''

    1- Séparer le texte encodé en groupes de 2 bytes
    2- Compiler les fréquences des groupes de 2 bytes dans une matrice (256 x 256) où:
        i = B1
        j = B2
        matrice[i,j] = nombre d'occurences de (B1 + B2)

        Exemple:
        Je vois 10 fois b1 + b2 dans le texte encodé.
        Alors matrice[b1, b2] = 10

    3- Trouver la même matrice mais pour la langue française (ex.: très très long texte), où au lieu de B1, B2 on a char1, char2

    Note:
        Dans un monde idéal:
            Sachant B1 = char1 (ex.: on connaît déjà B1 = "e ")
                On fait une recherche de max(matrice_encode[B1]) = B2.
                    On ne connaît pas B2
                On fait une recherche de max(matrice_fr[char1]) = char2.
                    Ça nous donne char2 = "le"
                
                B2 = char2 = "le" car les deux matrices nous donnent le plus fréquent après un certain caractère.
            Or, ceci est pour un monde idéal où les probabilité "match" 1 à 1. On n'est pas dans ce monde idéal.
            Qu'est-ce qu'on pourrait faire pour approximer ce monde idéal?

        
    4- Algorithme pour trouver des substitutions fiables automatiquement:
        i) Pour chaque byte connu B1 = char1:
            Trouver max(matrice[B1]) et 2e_max(matrice[B1])
                Ex.: max1 = B2
                Ex.: max2 = B3
            Trouver max(matrice_fr[char1]) et 2e_max(matrice_fr[char1])
                Ex.: max_fr1 = "qu"
                Ex.: max_fr2 = "un"
        
        ii) On va déterminer un seuil à partir duquel on peut dire que la substitution est fiable, donc B2 -> "qu"
            Prenons max1, max2, max_fr1, max_fr2:
                On accepte la substitution à partir de 20 occurences de B1

                On accepte la substitution à partir d'une différence de 2x entre max1 et max2
                max1 > 2*max2 et max_fr1 >= 2*max_fr2

        iii) Si on fait une substitution, on recommence au point i)
            
    5- Quand on sort de la boucle...?
        - Si on connaît plusieurs substitutions avec espaces:
            ex.: 'x '
                Regarder les bytes après 'x ' dans le texte encodé.
                Chaque byte après 'x ' ne commence pas par un espace.
            On peut donc avoir un ensemble 'no_starting_space'

            inversement:
            ex.: ' x'
                Regarder les bytes avant ' x' dans le texte encodé.
                Chaque byte avant ' x' ne termine pas par un espace.
            On peut donc avoir un ensemble 'no_ending_space'

            Si on connaît assez de 'x ' et ' x', on peut obtenir:
                no_starting space ∩ no_ending_space = bytes_no_spaces
            On peut donc obtenir un ensemble de bytes *sans* espaces.


        - Augmenter la dimension de la matrice et recommencer? (B1 -> B2) -> (B1 + B2 -> B3)   
    '''

    # ---------------------- 4. Substitution automatique des bytes ----------------------
    # Exemple pour obtenir la matrice de dimension 1:
    fr_text = sequence_fr_text("https://www.gutenberg.org/cache/epub/68355/pg68355.txt", "https://www.gutenberg.org/cache/epub/42131/pg42131.txt")
    (matrice_bytes, matrice_fr) = frequence_matrixes(dimension=1, fr_text=fr_text, encoded_bytes=encoded_bytes)


    """ # Écrire le fichier décodé
    decoded_text = decode(encoded_bytes, key)
    with open("decoded_text.txt", "w", encoding="utf-8") as f:
        f.write(decoded_text) """

    return






if __name__ == '__main__':

    main()
