import bisect
import os
import random as rnd
import re
import requests
from collections import Counter
import time


global allow_rand
allow_rand = True

global possible_chars_in_word
possible_chars_in_word = ['b', 'j', 'J', 'Â', 'É', 'ê', 't', 'Y', 'N', 'B', 'V', 'Ê',
                '’', 'i', 's', 'C', 'â', 'ï', 'W', 'y', 'p', 'D', 'A', 'n', 'q',
                'e', 'T', 'È', 'U', 'v', 'l', 'P', 'X', 'Z', 'À', 'ç', 'u', 'î', 'L', 'k', 'E', 'R',
                'é', 'O', 'Î', 'a', 'F', 'H', 'c', "'", 'è', 'I', 'S', 'x', 'à', 'g', 'Q', 'w', 'û', 'G', 'm', 'K', 'z', 'o', 'ù', 'r',
                'M', 'Ç', 'h', 'f', 'ë', 'd', 'ô', 'es',
                'en', 'qu', 're', 'de', 'le', 'nt', 'on', 'ou', 'ue',
                'an', 'te', 'ai', 'se', 'it', 'me', 'is', 'oi', 'er', 'ce', 'ne', 'et', 'in', 'ns',
                'ur', 'eu', 'co', 'tr', 'la', 'ar', 'ie', 'ui', 'us', 'ut', 'il', 'pa', 'au',
                'el', 'ti', 'st', 'un', 'em', 'ra', 'e,', 'so', 'or', 'll', 'nd', 'si', 'ir', 
                'ss', 'po', 'ro', 'ri', 'pr', 's,', 'ma', 'di', 'vo', 'pe', 'to', 'ch',
                've', 'nc', 'om', 'je', 'no', 'rt', 'lu', "'e", 'mo', 'ta', 'as', 'at', 'io', 'sa',
                "u'", 'av', 'os', "l'", "'a", 'rs', 'pl', 'ho', 'té', 'ét', 'fa', 'da', 'li',
                'su', 'ée', 'ré', 'dé', 'ec', 'nn', 'mm', "'i", 'ca', 'uv', 'id', 'ni', 'bl']

global fr_dict
fr_dict = set()

def capture_groups(start_str, n_bytes_min, n_bytes_max, end_str, encoded_bytes, key, decrypt_when_possible=False,
                       bytes_with_starting_spaces=[False] * 256, bytes_with_ending_spaces=[False] * 256,
                       include_start_and_end=False):
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
        for i in range(2, len(encoded_bytes) - 2):
            debug = False
            byte = encoded_bytes[i]
            next_byte = encoded_bytes[i + 1]
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
            if debug: print("i:", i, "group:", (group), "byte:", byte, "next_byte:", next_byte, "char1", char, "char2",
                            char2)
        return groups

def verif_key(key, secret_key):
    '''
    Fonction de débug pour vérifier si la clé générée est correcte.
    key = clé générée (byte -> char)
    secret_key = clé secrète (char -> byte)
    '''
    for byte in range(256):
        if key[byte] is None:
            continue
        if key[byte] != secret_key[byte]:
            print("Clé: ", key[byte], " | Secrète: ", secret_key[byte])

def encode_with_sequential_bytes(sequenced_text):
    '''
    Encodage du texte séquencé en bytes.
    Transforme chacun des caractères en bytes séquentiels.
    Retourne l'encodage en bytes et le dictionnaire de correspondance.
    '''
    byte_encode = 0
    chars_encoded = {}
    sequenced_bytes_text = []
    all_possible_chars = text_to_symbols(sequenced_text)
    for i in range(len(sequenced_text)):
        char = sequenced_text[i]
        if not char in all_possible_chars:
            continue
        char = char.replace('\r', '&').replace('\n', '@').replace('\ufeff', '<')
        
        if char not in chars_encoded:
            chars_encoded[char] = byte_encode
            byte_encode += 1
        sequenced_bytes_text.append(chars_encoded[char])

    # Inverser le dictionnaire chars_encoded pour obtenir byte -> char
    byte_to_char = {v: k for k, v in chars_encoded.items()}

    return sequenced_bytes_text, byte_to_char, chars_encoded


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

    sequenced_text, bytes_to_char, char_to_byte = encode_with_sequential_bytes(sequenced_text)

    return sequenced_text, bytes_to_char, char_to_byte

# Fonction pour obtenir le max et le 2e max d'une liste ainsi que leurs valeurs
def max_2_ignore(lst, ignore_set):
    max1, max2 = -1, -1
    max1_index, max2_index = -1, -1

    for i, value in enumerate(lst):
        if i in ignore_set:
            continue

        if value > max1:
            max2 = max1
            max2_index = max1_index
            max1 = value
            max1_index = i
        elif value > max2:
            max2 = value
            max2_index = i

    return max1, max2, max1_index, max2_index

def frequence_matrixes_combinations(dimension, encoded_bytes, add_indexes=False):
    """
    Crée une matrice de fréquences de bytes de dimension donnée.
    Problème:
        Étant donné N bytes fixes, déterminer la fréquence d'apparition de chaque byte formant un N+1-uplet avec ces N bytes dans le texte.
        Stocker les fréquences dans une matrice où chaque ligne représente un N-uplet et chaque colonne représente un byte.
            Modif: La colonne qui représente un byte est modifiée pour être un tuple. 
                À l'indexe 0 on a le byte, et à l'indexe 1 on a un tableau des indexes de toutes les occurences du groupe de bytes dans le texte

    Arguments:
        dimension: La dimension de la matrice voulue (nombre de bytes fixes dans les tuples)
        encoded_bytes: Le texte encodé sous forme de tableau de bytes
        byte_probabilities: Dictionnaire des probabilités pour chaque byte.
            Utilisé pour conserver l'ordre décroissant de probabilités des bytes pour l'insertion dans la matrice


    Retourne:
        La matrice de fréquences des bytes
    """
    if add_indexes:
        matrice = {}
        for i in range(len(encoded_bytes) - dimension):
            if dimension == 1:
                if encoded_bytes[i+1] in matrice:
                    liste_freq_bytes, indexes = matrice[encoded_bytes[i+1]]
                    liste_freq_bytes[encoded_bytes[i]] += 1
                    indexes.append(i)
                else:
                    liste_freq_bytes = [0 for _ in range(256)]
                    liste_freq_bytes[encoded_bytes[i]] += 1
                    indexes = [i]
                    matrice[encoded_bytes[i+1]] = (liste_freq_bytes, indexes)
                if encoded_bytes[i] in matrice:
                    liste_freq_bytes, indexes = matrice[encoded_bytes[i]]
                    liste_freq_bytes[encoded_bytes[i+1]] += 1
                    indexes.append(i)
                else:
                    liste_freq_bytes = [0 for _ in range(256)]
                    liste_freq_bytes[encoded_bytes[i+1]] += 1
                    indexes = [i]
                    matrice[encoded_bytes[i+1]] = (liste_freq_bytes, indexes)
            else:
                window = encoded_bytes[i:i + dimension + 1]
                
                for j in range(len(window)):
                    current_byte = window[j]
                    other_bytes = window[:j] + window[j + 1:]
                    other_bytes = sorted(other_bytes)
                    other_bytes = tuple(other_bytes)
                    
                    if other_bytes in matrice:
                        liste_freq_bytes, indexes = matrice[other_bytes]
                        liste_freq_bytes[current_byte] += 1
                        indexes.append(i)
                    else:
                        liste_freq_bytes = [0 for _ in range(256)]
                        liste_freq_bytes[current_byte] += 1
                        indexes = [i]
                        matrice[other_bytes] = (liste_freq_bytes, indexes)
    else:
        matrice = {}
        for i in range(len(encoded_bytes) - dimension):
            if dimension == 1:
                if encoded_bytes[i+1] in matrice:
                    matrice[encoded_bytes[i+1]][encoded_bytes[i]] += 1
                else:
                    matrice[encoded_bytes[i+1]] = [0 for _ in range(256)]
                    matrice[encoded_bytes[i+1]][encoded_bytes[i]] += 1
                if encoded_bytes[i] in matrice:
                    matrice[encoded_bytes[i]][encoded_bytes[i + 1]] += 1
                else:
                    matrice[encoded_bytes[i]] = [0 for _ in range(256)]
                    matrice[encoded_bytes[i]][encoded_bytes[i + 1]] += 1
            else:
                window = encoded_bytes[i:i + dimension + 1]
                
                for j in range(len(window)):
                    current_byte = window[j]
                    other_bytes = window[:j] + window[j + 1:]
                    other_bytes = sorted(other_bytes)
                    other_bytes = tuple(other_bytes)
                    
                    if other_bytes in matrice:
                        matrice[other_bytes][current_byte] += 1
                    else:
                        matrice[other_bytes] = [0 for _ in range(256)]
                        matrice[other_bytes][current_byte] += 1
    return matrice

def frequence_matrixes(dimension, fr_text, encoded_bytes):
    """
    Crée deux matrices de fréquences (bytes et langue française) de dimension donnée.
    Pour dimension = 1: les matrices sont des dictionnaires simples.
        Ex.: Pour obtenir la liste des bytes qui forment une paire avec le byte 1 on fait matrice[1]
    Pour dimension >1, les matrices sont en réalité des dictionnaires de tuples.
        Ex.: Pour obtenir la liste des bytes qui forment un trio avec le byte 1 et 3 on fait matrice_bytes[(1,3)]

    Arguments:
        dimension: La dimension de la matrice voulue
        texte_fr: Un tableau d'un texte en français dont les caractères sont séquencés selon l'encodage en bytes
        encoded_bytes: Les bytes encodés
    """
    matrice_bytes = {}
    matrice_fr = {}
    for texte, matrice in ((encoded_bytes, matrice_bytes), (fr_text, matrice_fr)):
        for i in range(len(texte) - dimension):
            window = texte[i:i + dimension]
            next_char_or_byte = texte[i + dimension]
            if dimension == 1:
                char_or_byte = window[0]
                if char_or_byte in matrice:
                    matrice[char_or_byte][next_char_or_byte] += 1
                else:
                    matrice[char_or_byte] = [0 for _ in range(256)]
                    matrice[char_or_byte][next_char_or_byte] += 1
            else:
                chars_or_bytes = tuple(window)
                if chars_or_bytes in matrice:
                    matrice[chars_or_bytes][next_char_or_byte] += 1
                else:
                    matrice[chars_or_bytes] = [0 for _ in range(256)]
                    matrice[chars_or_bytes][next_char_or_byte] += 1

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


def chiffrer(M, K):
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
    l = len(symboles)
    if l > 256:
        return False

    rnd.seed(1337)
    int_keys = rnd.sample(list(range(l)), l)
    dictionary = dict({})
    for s, k in zip(symboles, int_keys):
        dictionary[s] = "{:08b}".format(k)
    return dictionary


def text_to_symbols(text):
    symboles = ['b', 'j', '\r', 'J', '”', ')', 'Â', 'É', 'ê', '5', 't', '9', 'Y', '%', 'N', 'B', 'V', '\ufeff', 'Ê',
                '?', '’', 'i', ':', 's', 'C', 'â', 'ï', 'W', 'y', 'p', 'D', '—', '«', 'º', 'A', '3', 'n', '0', 'q', '4',
                'e', 'T', 'È', '$', 'U', 'v', '»', 'l', 'P', 'X', 'Z', 'À', 'ç', 'u', '…', 'î', 'L', 'k', 'E', 'R', '2',
                '_', '8', 'é', 'O', 'Î', '‘', 'a', 'F', 'H', 'c', '[', '(', "'", 'è', 'I', '/', '!', ' ', '°', 'S', '•',
                '#', 'x', 'à', 'g', '*', 'Q', 'w', '1', 'û', '7', 'G', 'm', '™', 'K', 'z', '\n', 'o', 'ù', ',', 'r',
                ']', '.', 'M', 'Ç', '“', 'h', '-', 'f', 'ë', '6', ';', 'd', 'ô', 'e ', 's ', 't ', 'es', ' d', '\r\n',
                'en', 'qu', ' l', 're', ' p', 'de', 'le', 'nt', 'on', ' c', ', ', ' e', 'ou', ' q', ' s', 'n ', 'ue',
                'an', 'te', ' a', 'ai', 'se', 'it', 'me', 'is', 'oi', 'r ', 'er', ' m', 'ce', 'ne', 'et', 'in', 'ns',
                ' n', 'ur', 'i ', 'a ', 'eu', 'co', 'tr', 'la', 'ar', 'ie', 'ui', 'us', 'ut', 'il', ' t', 'pa', 'au',
                'el', 'ti', 'st', 'un', 'em', 'ra', 'e,', 'so', 'or', 'l ', ' f', 'll', 'nd', ' j', 'si', 'ir', 'e\r',
                'ss', 'u ', 'po', 'ro', 'ri', 'pr', 's,', 'ma', ' v', ' i', 'di', ' r', 'vo', 'pe', 'to', 'ch', '. ',
                've', 'nc', 'om', ' o', 'je', 'no', 'rt', 'à ', 'lu', "'e", 'mo', 'ta', 'as', 'at', 'io', 's\r', 'sa',
                "u'", 'av', 'os', ' à', ' u', "l'", "'a", 'rs', 'pl', 'é ', '; ', 'ho', 'té', 'ét', 'fa', 'da', 'li',
                'su', 't\r', 'ée', 'ré', 'dé', 'ec', 'nn', 'mm', "'i", 'ca', 'uv', '\n\r', 'id', ' b', 'ni', 'bl']
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
        urls = ["https://www.gutenberg.org/ebooks/13846.txt.utf-8",
                "https://www.gutenberg.org/ebooks/4650.txt.utf-8",
                "https://www.gutenberg.org/cache/epub/35064/pg35064.txt",
                "https://www.gutenberg.org/cache/epub/14793/pg14793.txt",
                "https://www.gutenberg.org/cache/epub/73384/pg73384.txt",
                "https://www.gutenberg.org/cache/epub/74455/pg74455.txt",
                "https://www.gutenberg.org/cache/epub/13951/pg13951.txt",
                "https://www.gutenberg.org/cache/epub/55501/pg55501.txt"]
        combined_text = ""
        for url in urls:
            text = load_text_from_web(url)
            if len(text) > 10000:
                text = text[5000:-5000]  # Enlever les 5000 premiers et derniers caractères
            else:
                continue
            text = text[text.index(' ') + 1:text.rindex(' ') + 1]  # Enlever les premiers et derniers caractères jusqu'au premier et dernier mot
            combined_text += text
    
        cle_secrete = gen_key(text_to_symbols(combined_text))
        C = chiffrer2(combined_text, cle_secrete)
        # Enlever tous les caractères non-binaires dans C
        C = re.sub(r"[^01]", "", C)

        # Sachant que le texte encodé est des 1 et des 0 où chaque tranche de 8 bits est un caractère

        # Déterminer le nombre de 0 à ajouter au début du texte pour obtenir un nombre de bits multiple de 8
        num_padding_bits = (8 - (len(C) % 8)) % 8

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
        urls = ["https://www.gutenberg.org/ebooks/13846.txt.utf-8",
                "https://www.gutenberg.org/ebooks/4650.txt.utf-8",
                "https://www.gutenberg.org/cache/epub/35064/pg35064.txt",
                "https://www.gutenberg.org/cache/epub/14793/pg14793.txt",
                "https://www.gutenberg.org/cache/epub/73384/pg73384.txt",
                "https://www.gutenberg.org/cache/epub/74455/pg74455.txt",
                "https://www.gutenberg.org/cache/epub/13951/pg13951.txt",
                "https://www.gutenberg.org/cache/epub/55501/pg55501.txt"]
        combined_text = ""
        for url in urls:
            text = load_text_from_web(url)
            if len(text) > 10000:
                text = text[5000:-5000]  # Enlever les 5000 premiers et derniers caractères
            else:
                continue
            text = text[text.index(' ') + 1:text.rindex(' ') + 1]  # Enlever les premiers et derniers caractères jusqu'au premier et dernier mot
            combined_text += text
        
        # Séparer le texte en mots
        words = combined_text.split()

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
    url1 = "https://www.gutenberg.org/cache/epub/18812/pg18812.txt"#"https://www.gutenberg.org/ebooks/13846.txt.utf-8"
    corpus1 = load_text_from_web(url1)

    # Charger le deuxième corpus et enlever les 10 000 premiers caractères
    url2 = "https://www.gutenberg.org/cache/epub/51632/pg51632.txt"#"https://www.gutenberg.org/ebooks/4650.txt.utf-8"
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
        l = a + b
        c = rnd.randint(0, len(corpus) - l)
        M = corpus[c:c + l]
    else:
        M = corpus[10000:270000]

    #print("Longueur du message à décoder: ", len(M))

    cle_secrete = gen_key(text_to_symbols(M))

    C = chiffrer2(M, cle_secrete)
    # Enlever tous les caractères non-binaires dans C
    C = re.sub(r"[^01]", "", C)

    # Décoder le texte chiffré
    decrypt(C, cle_secrete)

    return


def decrypt(encoded_text, cle_secrete):
    # ---------------------- Préparation des données ----------------------

    # On utilise la clé secrète à des fins de tests
    secret_key_byte_to_char = {}
    for char, byte in cle_secrete.items():
        secret_key_byte_to_char[int(byte, 2)] = char.replace('\r', '&').replace('\n', '@').replace('\ufeff', '<')

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
    num_padding_bits = (8 - (len(encoded_text) % 8)) % 8

    # Ajouter les bits de padding au début du texte
    encoded_text = "0" * num_padding_bits + encoded_text

    # Séparer le texte encodé en bytes (convertis en entiers)
    encoded_bytes = [int(encoded_text[i:i + 8], 2) for i in range(0, len(encoded_text), 8)]

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
    global key
    key = [None] * 256
    # ______________ 1. Trouver la substitution "e " et "s " ___________

    # Première remarque: parfois dans le texte on tombe sur une page des matières, où il y a énormément de " " consécutifs pour de la mise en page.
    # On va lire les bytes encodés et cibler tous les bytes qui sont répétés 4+ fois consécutivement. Le byte le plus fréquent est l'espace.
    # Il est aussi très possible qu'on ne trouve aucun byte répété 4+ fois si on n'est pas sur une table des matières.
    groups_3_consecutive = []
    for i in range(len(encoded_bytes)):
        group = [encoded_bytes[i]]
        while i < len(encoded_bytes) - 1 and encoded_bytes[i] == encoded_bytes[i + 1]:
            i+=1
            group.append(encoded_bytes[i])
        if len(group) >= 3:
            groups_3_consecutive.append(group)
    
    # On veut savoir combien de groupes de 10+ consécutifs il y a pour chaque byte
    byte_to_10_consecutive_frequency = {}
    # On veut savoir le maximum de bytes consécutifs pour chaque byte
    byte_to_max_consecutive_len = {}
    for group in groups_3_consecutive:
        byte = group[0]
        if byte not in byte_to_max_consecutive_len or byte_to_max_consecutive_len[byte] < len(group):
            byte_to_max_consecutive_len[byte] = len(group)
        if byte not in byte_to_10_consecutive_frequency and len(group) >= 10:
            byte_to_10_consecutive_frequency[byte] = 1
        elif len(group)>=10:
            byte_to_10_consecutive_frequency[byte] += 1
    
    # On trouve le byte avec le plus de bytes consécutifs
    max_len = 0
    max_byte = -1
    for byte, length in byte_to_max_consecutive_len.items():
        if length > max_len:
            max_len = length
            max_byte = byte

    # On prend la liste des 3 bytes les plus fréquents dans byte_to_max_consecutive_len
    sorted_max_consecutive_len_bytes = sorted(byte_to_max_consecutive_len, key=byte_to_max_consecutive_len.get, reverse=True)
    # On prend les 3 bytes les plus fréquents
    top_3_bytes = sorted_max_consecutive_len_bytes[:3]
    
    # On suppose que si max_len > 10, que le byte en question est dans le top 3 bytes avec le plus de répétitions, 
    # et qu'il apparaît au moins 3 fois dans le texte avec un groupe de 10+ consécutif, c'est un espace
    if max_len > 10 and max_byte in top_3_bytes and byte_to_10_consecutive_frequency[max_byte] > 3:
        key[max_byte] = ' '


    # 1- On suppose que les 2 bytes les plus probables sont "e " et "s ".
    # Il est possible que "\r\n" soit premier ou 2e, on va traiter ce cas particulier aussi.
    #   Note: "\r\n" est transformé en "&@" pour éviter les problèmes de lecture

    # On doit trouver lequel est "e " et lequel est "s "
    # On va supposer que le premier est "e ", et le 2e est "s ".
    if key[sorted_bytes[0]] == ' ':
        e_index = 1
        s_index = 2
    elif key[sorted_bytes[1]] == ' ':
        e_index = 0
        s_index = 2
    else:
        e_index = 0
        s_index = 1
        
    key[sorted_bytes[e_index]] = 'e '
    key[sorted_bytes[s_index]] = 's '

    # On forme des groupes de mots de forme "e " + BYTE + "e " et des groupes de "s " + BYTE + "s "
    # L'objectif est que les groupes de forme "e " + BYTE + "e " ont moins de BYTE possibles
    # et que les groupes de forme "s " + BYTE + "s " ont beaucoup de BYTE possibles
    groups_e = capture_groups(start_str='e ', n_bytes_min=1, n_bytes_max=1, end_str='e ', encoded_bytes=encoded_bytes, key=key)
    groups_s = capture_groups(start_str='s ', n_bytes_min=1, n_bytes_max=1, end_str='s ', encoded_bytes=encoded_bytes, key=key)

    # On vérifie que la taille des groupes est >= 5:
    # Si non, ça veut dire que le groupe associé est en réalité "\r\n",
    #   car il ne devrait pas (ou très peu) y avoir 2 sauts de ligne consécutifs séparés d'un seul byte dans le texte.
    if len(groups_e) < 5:
        # Le 'e ' choisi est en réalité \n\r
        key[sorted_bytes[e_index]] = '&@'
        e_index += 1
        if key[sorted_bytes[e_index]] == ' ': # Prendre en compte le cas de l'espace déjà assigné
            e_index += 1
        key[sorted_bytes[e_index]] = 'e '

        s_index += 1
        if key[sorted_bytes[s_index]] == ' ': # Prendre en compte le cas de l'espace déjà assigné
            s_index += 1
        key[sorted_bytes[s_index]] = 's '
        #print("Étape 1: Substitution de '" + str(sorted_bytes[0]) + "' par '&@', Obs = " + str(byte_probabilities[sorted_bytes[0]]) + " | Théorique = " + str(char_probabilities['&@']))
        #print("Validation avec la clé secrète: '" + secret_key_byte_to_char[sorted_bytes[1]] + "'")
        groups_e = capture_groups(start_str='e ', n_bytes_min=1, n_bytes_max=1, end_str='e ', encoded_bytes=encoded_bytes, key=key)
        groups_s = capture_groups(start_str='s ', n_bytes_min=1, n_bytes_max=1, end_str='s ', encoded_bytes=encoded_bytes, key=key)
    elif len(groups_s) < 5:
        # Le \n\r est le 2e caractère le plus fréquent de notre texte.
        key[sorted_bytes[e_index]] = 'e '
        key[sorted_bytes[s_index]] = '&@'
        s_index += 1
        if key[sorted_bytes[s_index]] == ' ': # Prendre en compte le cas de l'espace déjà assigné
            s_index += 1
        key[sorted_bytes[s_index]] = 's '
        #print("Étape 1: Substitution de '" + str(sorted_bytes[1]) + "' par '&@', Obs = " + str(byte_probabilities[sorted_bytes[1]]) + " | Théorique = " + str(char_probabilities['&@']))
        #print("Validation avec la clé secrète: '" + secret_key_byte_to_char[sorted_bytes[1]] + "'")
        groups_e = capture_groups(start_str='e ', n_bytes_min=1, n_bytes_max=1, end_str='e ', encoded_bytes=encoded_bytes, key=key)
        groups_s = capture_groups(start_str='s ', n_bytes_min=1, n_bytes_max=1, end_str='s ', encoded_bytes=encoded_bytes, key=key)
    
    # Maintenant qu'on a traité le cas particulier où "\r\n" est premier ou 2e, on peut continuer en discernant "e " et "s "
    # On calcule les fréquences de chaque byte dans les groupes de "e "
    
    #print("Groups_e:")
    # Créer un set avec tous les bytes dans groups_e:
    all_bytes_e = []
    for group in groups_e:
        for byte in group:
            if byte not in all_bytes_e:
                #print(secret_key_byte_to_char[byte], end=" ")
                all_bytes_e.append(byte)


    #print("\nGroups_s:")
    # Créer un set avec tous les bytes dans groups_s:
    all_bytes_s = []
    for group in groups_s:
        for byte in group:
            if byte not in all_bytes_s:
                #print(secret_key_byte_to_char[byte], end=" ")
                all_bytes_s.append(byte)

    #print("Nombre de bytes différents dans les groupes de 'e ': ", len(all_bytes_e))
    #print("Nombre de bytes différents dans les groupes de 's ': ", len(all_bytes_s))

    # Le groupe avec le plus de bytes différents est "s "
    # Appliquer l'échange si on s'est trompé dans l'assignation
    if len(all_bytes_e) > len(all_bytes_s):
        key[sorted_bytes[e_index]] = 's '
        key[sorted_bytes[s_index]] = 'e '
        

    #print("Étape 1: Substitution de '" + str(sorted_bytes[e_index]) + "' par 'e ', Obs = " + str(byte_probabilities[sorted_bytes[e_index]]) + " | Théorique = " + str(char_probabilities['e ']))
    #print("Validation avec la clé secrète: '" + secret_key_byte_to_char[sorted_bytes[e_index]] + "'")
    #print("Étape 1: Substitution de '" + str(sorted_bytes[s_index]) + "' par 's ', Obs = " + str(byte_probabilities[sorted_bytes[s_index]]) + " | Théorique = " + str(char_probabilities['s ']))
    #print("Validation avec la clé secrète: '" + secret_key_byte_to_char[sorted_bytes[s_index]] + "'")

    """ # Print les 10 autres bytes les plus probables ainsi que la clé secrète pour déboggage
    print("10 bytes les plus probables: clé-dictionnaire vs clé secrète")
    for i in range(0,10):
        print("Char : '" + sorted_chars[i] + "' =  char '" + secret_key_byte_to_char[sorted_bytes[i]] +"', Obs = " + str(byte_probabilities[sorted_bytes[i]]) + " | Théorique = " + str(char_probabilities[sorted_chars[i]]))
    """

    # ------------------------- TOUS LES BYTES QUI SUIVENT 'e ' -------------------------

    # Les bytes qui suivent "e " dans le texte. Ces bytes sont garantis de ne pas commencer par un espace.
    no_starting_space = []
    prev_byte = None
    for byte in encoded_bytes:
        if key[byte] is None and prev_byte is not None and key[prev_byte] == 'e ':
            if byte not in no_starting_space:
                no_starting_space.append(byte)
        prev_byte = byte

    # print("\nVoici les bytes qui suivent 'e ' dans le texte ainsi que leurs fréquences observées vs théoriques:")
    no_starting_space_pairs = []
    for byte in no_starting_space:
        no_starting_space_pairs.append((byte, byte_probabilities[byte]))

    # Trier les pairs en ordre décroissant de probabilité
    no_starting_space_pairs = sorted(no_starting_space_pairs, key=lambda x: x[1], reverse=True)
    i = 0
    for (byte, prob) in no_starting_space_pairs:
        i += 1
        # print(str(i) + ": '" + secret_key_byte_to_char[byte] + "' Obs: " + str(prob) + " | Théorique: " + str(char_probabilities[secret_key_byte_to_char[byte]]))

    # Obtenir les probabilités des bytes qui commencent peut-être par " " en inversant no_starting_space
    maybe_starting_space = []
    for byte in byte_probabilities:
        if not byte in no_starting_space:
            maybe_starting_space.append(byte)

    # print("\nVoici les bytes qui ne suivent jamais 'e ' dans le texte ainsi que leurs fréquences observées vs théoriques:")
    maybe_starting_space_pairs = []
    for byte in maybe_starting_space:
        maybe_starting_space_pairs.append((byte, byte_probabilities[byte]))

    # Trier les bytes en ordre décroissant de probabilité
    maybe_starting_space_pairs = sorted(maybe_starting_space_pairs, key=lambda x: x[1], reverse=True)
    i = 0
    for (byte, prob) in maybe_starting_space_pairs:
        i += 1
        # print(str(i) + ": '" + secret_key_byte_to_char[byte] + "' Obs: " + str(prob) + " | Théorique: " + str(char_probabilities[secret_key_byte_to_char[byte]]))

    # ------------------------- TOUS LES BYTES QUI PRÉCÈDENT 'e ' -------------------------
    # Les bytes qui précèdent "e " dans le texte. Ces bytes sont garantis de ne pas se terminer par un espace, car " e " n'est pas un mot.
    no_ending_space = []
    prev_byte = None
    for byte in encoded_bytes:
        if key[byte] is not None and key[byte] == 'e ' and prev_byte is not None:
            if prev_byte not in no_ending_space:
                no_ending_space.append(prev_byte)
        prev_byte = byte

    # print("\nVoici les bytes qui précèdent 'e ' dans le texte ainsi que leurs fréquences observées vs théoriques:")
    no_ending_space_pairs = []
    for byte in no_ending_space:
        no_ending_space_pairs.append((byte, byte_probabilities[byte]))

    # Trier les pairs en ordre décroissant de probabilité
    no_ending_space_pairs = sorted(no_ending_space_pairs, key=lambda x: x[1], reverse=True)
    i = 0
    for (byte, prob) in no_ending_space_pairs:
        i += 1
        # print(str(i) + ": '" + secret_key_byte_to_char[byte] + "' Obs: " + str(prob) + " | Théorique: " + str(char_probabilities[secret_key_byte_to_char[byte]]))

    # Obtenir les probabilités des bytes qui ne précèdent jamais 'e '
    maybe_ending_space = []
    for byte in byte_probabilities:
        if not byte in no_ending_space:
            maybe_ending_space.append(byte)

    # print("\nVoici les bytes qui ne précèdent jamais 'e ' dans le texte ainsi que leurs fréquences observées vs théoriques:")
    maybe_ending_space_pairs = []
    for byte in maybe_ending_space:
        maybe_ending_space_pairs.append((byte, byte_probabilities[byte]))

    # Trier les pairs en ordre décroissant de probabilité
    maybe_ending_space_pairs = sorted(maybe_ending_space_pairs, key=lambda x: x[1], reverse=True)
    i = 0
    for (byte, prob) in maybe_ending_space_pairs:
        i += 1
        # print(str(i) + ": '" + secret_key_byte_to_char[byte] + "' Obs: " + str(prob) + " | Théorique: " + str(char_probabilities[secret_key_byte_to_char[byte]]))

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
    groups = capture_groups(start_str=r'e ', n_bytes_min=1, n_bytes_max=1, end_str=r'e ', encoded_bytes=encoded_bytes, key=key)

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
    #print("Étape 3: Substitution de '" + str(sorted_occurences_group[0]) + "' par 'qu'")
    #print("Validation avec la clé secrète: '" + secret_key_byte_to_char[sorted_occurences_group[0]] + "'")

    # On cherche maintenant tous les bytes qui suivent "e " + "qu" sauf ceux déjà décodés
    bytes_after_e_qu = {}
    for i in range(2, len(encoded_bytes)):
        byte = encoded_bytes[i]
        prev_prev_char = key[encoded_bytes[i - 2]]
        prev_char = key[encoded_bytes[i - 1]]
        if prev_prev_char == 'e ' and prev_char == 'qu' and key[byte] == None:
            if byte in bytes_after_e_qu:
                bytes_after_e_qu[byte] += 1
            else:
                bytes_after_e_qu[byte] = 1

    # Trier les bytes en ordre décroissant d'occurences
    sorted_bytes_after_e_qu = sorted(bytes_after_e_qu, key=bytes_after_e_qu.get, reverse=True)

    # Imprimer les premiers bytes après "e " + "qu" avec leur probabilité observée vs réelle
    # print("\nBytes après 'e ' + 'qu':")
    i = 0
    for byte in sorted_bytes_after_e_qu:
        i += 1
        # print(str(i) + ": Freq.: " + str(bytes_after_e_qu[byte]) + " | '" + secret_key_byte_to_char[byte] + "' Obs: " + str(byte_probabilities[byte]) + " | Théorique: " + str(char_probabilities[secret_key_byte_to_char[byte]]))


    groups = capture_groups(start_str=r'(e |i )', n_bytes_min=1, n_bytes_max=1, end_str=r'(e |i )',
                            include_start_and_end=True, encoded_bytes=encoded_bytes, key=key)
    # print("Nombre de groupes capturés: ", len(groups))
    for group in groups:
        group_str = ""
        for byte in group:
            # Vérifier si byte est une String
            if isinstance(byte, str):
                group_str += byte
            else:
                group_str += secret_key_byte_to_char[byte]
        # print("Mot '" + group_str + "'")

    verif_key(key, secret_key_byte_to_char)

    # ---------------------- 4. Substitution automatique des bytes ----------------------
    # 1- Encoder un texte en français qui servira de données théoriques pour les fréquences des caractères.
    # fr_text: Texte en français encodé en bytes
    fr_text, fr_byte_to_char, fr_char_to_byte = sequence_fr_text("https://www.gutenberg.org/ebooks/13846.txt.utf-8",
                                                                 "https://www.gutenberg.org/ebooks/4650.txt.utf-8",
                                                                 "https://www.gutenberg.org/cache/epub/35064/pg35064.txt",
                                                                 "https://www.gutenberg.org/cache/epub/14793/pg14793.txt",
                                                                 "https://www.gutenberg.org/cache/epub/73384/pg73384.txt",
                                                                 "https://www.gutenberg.org/cache/epub/74455/pg74455.txt",
                                                                 "https://www.gutenberg.org/cache/epub/13951/pg13951.txt",
                                                                 "https://www.gutenberg.org/cache/epub/55501/pg55501.txt")

    '''
    2- Compiler les fréquences des groupes de 2 bytes dans une matrice (256 x 256) où:
        i = B1
        j = B2
        matrice[i,j] = nombre d'occurences de (B1 + B2)

        Exemple:
        Je vois 10 fois b1 + b2 dans le texte encodé.
        Alors matrice[b1, b2] = 10

    3- Compiler la même chose pour la langue française où matrice_fr[char1, char2] = nombre d'occurences de (char1 + char2)
    '''

    '''
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
    '''

    # Initialiser les bytes connus
    bytes_connus = [byte for byte in range(256) if key[byte] is not None]
    bytes_connus = sorted(bytes_connus)
    bytes_connus_autre_encodage = [fr_char_to_byte[key[byte]] for byte in bytes_connus]
    bytes_connus_autre_encodage = sorted(bytes_connus_autre_encodage)

    # Fonction pour performer les substitutions
    def perform_substitution(matrice_bytes, matrice_fr, dimension, min_freq, ratio_threshold_bytes, ratio_threshold_fr, char_probabilities, byte_probabilities, excluded_substitutions, exclude_substitutions, encoded_bytes):

        for (context, freqs_bytes) in matrice_bytes.items():
            # Vérifie si le contexte consiste en des bytes connus
            if dimension == 1:
                byte = context
                if byte not in bytes_connus:
                    continue
                char = key[byte]
                # Obtenir le contexte correspondant dans le texte français
                fr_context = fr_char_to_byte.get(char)
                if fr_context is None or fr_context not in matrice_fr:
                    continue
                freqs_fr = matrice_fr[fr_context]
            else:
                if not all(b in bytes_connus for b in context):
                    continue
                # Pour dimension > 1, 'contexte' est un tuple de bytes
                # Obtenir le contexte de caractères
                char_context = tuple(key[b] for b in context)
                # Convertir le contexte de caractères en bytes dans l'encodage français
                byte_context_fr = tuple(sorted(fr_char_to_byte[ch] for ch in char_context))
                # Obtenir les fréquences des bytes suivant le contexte dans le texte français
                freqs_fr = matrice_fr.get(byte_context_fr)
                if freqs_fr is None:
                    continue

            # Ignorer les bytes déjà connus
            freqs_bytes_ignore = freqs_bytes
            freqs_fr_ignore = freqs_fr

            # Trouver les bytes les plus fréquents dans les deux fréquences
            max1, max2, max1_index, max2_index = max_2_ignore(freqs_bytes_ignore, bytes_connus)
            max1_fr, max2_fr, max1_fr_index, max2_fr_index = max_2_ignore(freqs_fr_ignore, bytes_connus_autre_encodage)

            # On a déjà exclu cette substitution
            if excluded_substitutions[max1_index][max1_fr_index]:
                continue

            # Obtenir les probabilités des 2 bytes les plus fréquents
            max1_prob = byte_probabilities.get(max1_index)
            max2_prob = byte_probabilities.get(max2_index)
            if max1_prob is None or max2_prob is None:
                continue
            

            # Obtenir la probabilité théorique des 2 char les plus fréquents
            max1_fr_prob = char_probabilities[fr_byte_to_char[max1_fr_index]]
            max2_fr_prob = char_probabilities[fr_byte_to_char[max2_fr_index]]

            # On veut s'assurer que la substitution max1 -> max1_fr est fiable
            # Pour ce faire, on évalue l'ordre relatif des probabilités de (max1, max2), et (max1_fr, max2_fr)
            #   Si (max1_prob > max_prob) & (max1_fr_prob > max2_fr_prob), on permet la substitution
            #   Si (max1_prob < max2_prob) & (max1_fr_prob < max2_fr_prob), on permet la substitution.
            # La substitution est bloquée uniquement dans le cas où l'ordre relatif entre les probabilités observées et théoriques sont inversées
            respects_order = (max1_prob > max2_prob) == (max1_fr_prob > max2_fr_prob)

            # Appliquer les seuils pour assurer des substitutions fiables
            if (
                    max1 >= min_freq
                    and max1 > ratio_threshold_bytes * max2
                    and max1_fr >= min_freq
                    and max1_fr > ratio_threshold_fr * max2_fr
            ):
                # Vérifier si la substitution est plausible
                if key[max1_index] is None and fr_byte_to_char[max1_fr_index] not in key:
                    # Bloquer la substitution si c'est le cas
                    if not respects_order:
                        #print("Substitution bloquée: " + secret_key_byte_to_char[max1_index] + " ou " + secret_key_byte_to_char[max2_index] + " -> " + fr_byte_to_char[max1_fr_index] + " ou " + fr_byte_to_char[max2_fr_index])
                        continue

                    # On fait une dernière vérification (très coûteuse) pour savoir si cette substitution est fiable.
                    if exclude_substitutions and not verify_substitution_in_text(max1_index, fr_byte_to_char[max1_fr_index], key, encoded_bytes):
                        print("Substitution bloquée: " + secret_key_byte_to_char[max1_index] + " -> " + fr_byte_to_char[max1_fr_index])
                        excluded_substitutions[max1_index][max1_fr_index] = True
                        continue



                    key[max1_index] = fr_byte_to_char[max1_fr_index]
                    bisect.insort(bytes_connus, max1_index)
                    bisect.insort(bytes_connus_autre_encodage, max1_fr_index)
                    # Append ce texte à un fichier "decoding.txt" pour vérification
                    with open("decoding.txt", "a", encoding="utf-8") as f:
                        f.write(f"'{fr_byte_to_char[max1_fr_index]}' = '{secret_key_byte_to_char[max1_index]}' with dimension '{str(dimension)}'\n")
                        f.write(f"min_freq={min_freq}, ratio_threshold_bytes={ratio_threshold_bytes}, ratio_threshold_fr={ratio_threshold_fr}\n")
                    return
        return

    # ------------------------------------------- ATTENTION --------------------------------------------------
    # À ceux qui seront assez braves pour toucher à ce code, sachez que la fonction perform_substitution est très sensible
    # L'auxiliaire me dit que la taille des textes de test sont sujet à changement, il faudra peut-être modifier ces paramètres pour être
    #   en fonction de la taille du message encodé.

    # ------------------ PARAMÈTRES POUR LES SÉQUENCES --------------------------
    initial_min_freq_seq = 15
    initial_ratio_threshold_bytes_seq = 1.7
    initial_ratio_threshold_fr_seq = 1.7

    min_min_freq_seq = 8
    min_ratio_threshold_bytes_seq = 1.3
    min_ratio_threshold_fr_seq = 1.3

    # Décrémenter les seuils
    min_freq_decrement_seq = 1
    ratio_threshold_decrement_seq = 0.1

    freq_decrement_per_dimension_seq = 0
    ratio_decrement_per_dimension_seq = 0

    # Dimensions des n-grammes à considérer
    dimensions_seq = [5, 4, 3, 2]

    matrices_bytes_seq = [None for _ in range(max(dimensions_seq))]
    matrices_fr_seq = [None for _ in range(max(dimensions_seq))]
    for dimension in dimensions_seq:
        matrices_bytes_seq[dimension-1], matrices_fr_seq[dimension-1] = frequence_matrixes(dimension=dimension, fr_text=fr_text, encoded_bytes=encoded_bytes)


    # --------------- PARAMÈTRES POUR LES COMBINAISONS ------------------------
    initial_min_freq_comb = 20
    initial_ratio_threshold_bytes_comb = 1.9
    initial_ratio_threshold_fr_comb = 1.9

    min_min_freq_comb = 12
    min_ratio_threshold_bytes_comb = 1.35
    min_ratio_threshold_fr_comb = 1.35


    # Décrémenter les seuils
    min_freq_decrement_comb = 1
    ratio_threshold_decrement_comb = 0.1

    freq_decrement_per_dimension_comb = 1
    ratio_decrement_per_dimension_comb = 0.1

    # Dimensions des n-grammes à considérer
    dimensions_comb = [1, 2]

    matrices_fr_comb = [None for _ in range(len(dimensions_comb))]
    matrices_bytes_comb = [None for _ in range(len(dimensions_comb))]
    for dimension in dimensions_comb:
        matrices_bytes_comb[dimension-1] = frequence_matrixes_combinations(dimension=dimension, encoded_bytes=encoded_bytes)
        matrices_fr_comb[dimension-1] = frequence_matrixes_combinations(dimension=dimension, encoded_bytes=fr_text)


    # --------------------- PARAMÈTRES GÉNÉRAUX -------------------------------
    # Nombre minimum de bytes connus désirés
    desired_minimum_known_bytes = len(byte_probabilities)

    # Après combien de substitutions de combinaisons on passe aux fréquences, et vice-versa
    # On veut diminuer le nombre de substitutions consécutives par la même approche pour éviter le plus possible les erreurs
    no_of_substitutions_comb = [desired_minimum_known_bytes]
    no_of_substitutions_seq = [5]
    swap_thresholds = []
    cumul = 0
    for i in range(len(no_of_substitutions_comb)):
        cumul += no_of_substitutions_comb[i]
        swap_thresholds.append(cumul)
        cumul += no_of_substitutions_seq[i]
        swap_thresholds.append(cumul)
    while cumul < desired_minimum_known_bytes + 1:
        cumul += no_of_substitutions_comb[-1]
        swap_thresholds.append(cumul)
        cumul += no_of_substitutions_seq[-1]
        swap_thresholds.append(cumul)


    swap_threshold_current_index = 0


    
    # --------------------- PRÉPARATION DE LA PREMIÈRE ITÉRATION --------------------------
    initial_min_freq = initial_min_freq_comb
    initial_ratio_threshold_bytes = initial_ratio_threshold_bytes_comb
    initial_ratio_threshold_fr = initial_ratio_threshold_fr_comb
    
    current_min_freq = initial_min_freq_comb
    current_ratio_threshold_bytes = initial_ratio_threshold_bytes_comb
    current_ratio_threshold_fr = initial_ratio_threshold_fr_comb

    min_min_freq = min_min_freq_comb
    min_ratio_threshold_bytes = min_ratio_threshold_bytes_comb
    min_ratio_threshold_fr = min_ratio_threshold_fr_comb

    min_freq_decrement = min_freq_decrement_comb
    ratio_threshold_decrement = ratio_threshold_decrement_comb

    freq_decrement_per_dimension = freq_decrement_per_dimension_comb
    ratio_decrement_per_dimension = ratio_decrement_per_dimension_comb

    dimensions = dimensions_comb

    matrices_bytes = matrices_bytes_comb
    matrices_fr = matrices_fr_comb

    nombre_bytes_connus_initial = len(bytes_connus)

    excluded_substitutions = [[False for _ in range(256)] for _ in range(256)]
    # À partir de combien de substitutions on fait la longue vérification de si la substitution est fiable
    exclude_substitutions_at = 10
    exclude_substitutions = False

    global possible_chars_in_word
    possible_chars_in_word = set(possible_chars_in_word)

    # Ouvrir le dictionnaire de la langue française "word_probabilities_cleansed.txt".
    global fr_dict
    fr_dict = set()
    with open("word_probabilities_cleansed.txt", "r", encoding="utf-8") as f:
        for word in f:
            fr_dict.add(word.strip())
            if word[-1].isalpha():
                # On considère qu'un mot peut se terminer par un point ou une virgule
                fr_dict.add(word + ",")
                fr_dict.add(word + ".")

    # Clear le fichier "decoding.txt"
    with open("decoding.txt", "w", encoding="utf-8") as f:
        f.write("------------- COMBINAISONS --------------\n")

    # -------------------- ITÉRATION DES SUBSTITUTIONS --------------------------
    while (
            current_min_freq >= min_min_freq
            and current_ratio_threshold_bytes >= min_ratio_threshold_bytes
            and current_ratio_threshold_fr >= min_ratio_threshold_fr
    ):
        substitutions_made = True
        while substitutions_made:
            substitutions_made = False
            for dimension in dimensions:
                #print(f"Début des substitutions avec n-grammes de dimension {dimension}")
                matrice_bytes = matrices_bytes[dimension - 1]
                
                matrice_fr = matrices_fr[dimension - 1]

                # Adjust thresholds based on dimension
                adjusted_min_freq = max(
                    current_min_freq - (dimension - 1) * freq_decrement_per_dimension,
                    min_min_freq
                )
                adjusted_ratio_threshold_bytes = max(
                    current_ratio_threshold_bytes - (dimension - 1) * ratio_decrement_per_dimension,
                    min_ratio_threshold_bytes
                )
                adjusted_ratio_threshold_fr = max(
                    current_ratio_threshold_fr - (dimension - 1) * ratio_decrement_per_dimension,
                    min_ratio_threshold_fr
                )

                #print(
                #    f"Dimension {dimension}: min_freq={adjusted_min_freq}, "
                #    f"ratio_threshold_bytes={adjusted_ratio_threshold_bytes}, "
                #    f"ratio_threshold_fr={adjusted_ratio_threshold_fr}"
                #)

                # Perform substitutions
                substitutions_before = len(bytes_connus)
                perform_substitution(
                                    matrice_bytes,
                                    matrice_fr,
                                    dimension,
                                    adjusted_min_freq,
                                    adjusted_ratio_threshold_bytes,
                                    adjusted_ratio_threshold_fr,
                                    char_probabilities,
                                    byte_probabilities,
                                    excluded_substitutions,
                                    exclude_substitutions,
                                    encoded_bytes
                                )
                substitutions_after = len(bytes_connus)
                if len(bytes_connus) - nombre_bytes_connus_initial >= exclude_substitutions_at:
                    exclude_substitutions = True
                if substitutions_after - nombre_bytes_connus_initial >= swap_thresholds[swap_threshold_current_index]:
                    break

                if substitutions_after > substitutions_before:
                    substitutions_made = True
                    # Recommencer depuis la première dimension pour prendre en compte les nouvelles substitutions
                    # Tout en remontant les seuils
                    current_min_freq = initial_min_freq
                    current_ratio_threshold_bytes = initial_ratio_threshold_bytes
                    current_ratio_threshold_fr = initial_ratio_threshold_fr
                    break
        # Vérifier si le nombre de bytes connus est suffisant
        if len(bytes_connus) >= desired_minimum_known_bytes:
            print(f"Nombre de symboles connus suffisant atteint: {len(bytes_connus)} symboles.")
            break
        else:
            if current_min_freq == min_min_freq and current_ratio_threshold_bytes == min_ratio_threshold_bytes and current_ratio_threshold_fr == min_ratio_threshold_fr:
                print("Seuils minimaux atteints, mais nombre de symboles connus insuffisant.")
                break

            # On arrête d'utiliser des combinaisons. Pour garder plus de précision, on passe aux séquences de bytes.
            if len(bytes_connus) - nombre_bytes_connus_initial >= swap_thresholds[swap_threshold_current_index]:
                if swap_threshold_current_index % 2 == 0:
                    if (swap_threshold_current_index//2)<len(no_of_substitutions_comb): 
                        print("Nombre de symboles connus par combinaisons a atteint le seuil de " + str(no_of_substitutions_comb[swap_threshold_current_index//2]) + ".")
                    with open("decoding.txt", "a", encoding="utf-8") as f:
                        f.write("----------------- SÉQUENCES -----------------\n")
                    # On passe aux séquences
                    initial_min_freq = initial_min_freq_seq
                    initial_ratio_threshold_bytes = initial_ratio_threshold_bytes_seq
                    initial_ratio_threshold_fr = initial_ratio_threshold_fr_seq

                    current_min_freq = initial_min_freq_seq
                    current_ratio_threshold_bytes = initial_ratio_threshold_bytes_seq
                    current_ratio_threshold_fr = initial_ratio_threshold_fr_seq

                    min_min_freq = min_min_freq_seq
                    min_ratio_threshold_bytes = min_ratio_threshold_bytes_seq
                    min_ratio_threshold_fr = min_ratio_threshold_fr_seq

                    # Décrémenter les seuils
                    min_freq_decrement = min_freq_decrement_seq
                    ratio_threshold_decrement = ratio_threshold_decrement_seq

                    freq_decrement_per_dimension = freq_decrement_per_dimension_seq
                    ratio_decrement_per_dimension = ratio_decrement_per_dimension_seq

                    # Dimensions des n-grammes à considérer
                    dimensions = dimensions_seq

                    # Utilisation de frequence_matrixes au lieu de frequence_matrixes_combinations
                    matrices_bytes = matrices_bytes_seq
                    matrices_fr = matrices_fr_seq

                    swap_threshold_current_index += 1
                else:
                    if (swap_threshold_current_index//2)<len(no_of_substitutions_seq):
                        print("Nombre de symboles connus par fréquences a atteint le seuil de " + str(no_of_substitutions_comb[swap_threshold_current_index//2]) + ".")
                    with open("decoding.txt", "a", encoding="utf-8") as f:
                        f.write("--------------- COMBINAISONS ----------------\n")
                    # On passe aux combinaisons
                    initial_min_freq = initial_min_freq_comb
                    initial_ratio_threshold_bytes = initial_ratio_threshold_bytes_comb
                    initial_ratio_threshold_fr = initial_ratio_threshold_fr_comb

                    current_min_freq = initial_min_freq_comb
                    current_ratio_threshold_fr = initial_ratio_threshold_fr_comb
                    current_ratio_threshold_bytes = initial_ratio_threshold_bytes_comb

                    min_min_freq = min_min_freq_comb
                    min_ratio_threshold_bytes = min_ratio_threshold_bytes_comb
                    min_ratio_threshold_fr = min_ratio_threshold_fr_comb

                    # Décrémenter les seuils
                    min_freq_decrement = min_freq_decrement_comb
                    ratio_threshold_decrement = ratio_threshold_decrement_comb

                    freq_decrement_per_dimension = freq_decrement_per_dimension_comb
                    ratio_decrement_per_dimension = ratio_decrement_per_dimension_comb

                    # Dimensions des n-grammes à considérer
                    dimensions = dimensions_comb

                    # Utilisation de frequence_matrixes au lieu de frequence_matrixes_combinations
                    matrices_bytes = matrices_bytes_comb
                    matrices_fr = matrices_fr_comb


                    swap_threshold_current_index += 1
            else:
                # Ajuster les seuils globaux
                current_min_freq = max(current_min_freq - min_freq_decrement, min_min_freq)
                current_ratio_threshold_bytes = max(
                    current_ratio_threshold_bytes - ratio_threshold_decrement, min_ratio_threshold_bytes
                )
                current_ratio_threshold_fr = max(
                    current_ratio_threshold_fr - ratio_threshold_decrement, min_ratio_threshold_fr
                )
                """ print(
                    f"Ajustement des seuils globaux: min_freq={current_min_freq}, "
                    f"ratio_threshold_bytes={current_ratio_threshold_bytes}, "
                    f"ratio_threshold_fr={current_ratio_threshold_fr}"
                ) """

    # -------------------------------- FINITION DU DÉCODAGE ------------------------------------
    # 1- Lance l'algorithme et regarde les print de la fonction verifier_substitution
    #       Tu verras qu'elle trouve parfois les bonnes substitutions avant notre algorithme principal en complétant les mots directement.
    #       Cependant elle ne choisit pas la substitution, elle fait seulement rejeter d'autres substitutions incorrectes.
    #       On va modifier la fonction pour qu'elle puisse choisir la substitution si elle est certaine. Je te laisse réfléchir à comment l'implémenter.
    # 2- Une fois l'itération avec les seuils terminée (parce qu'elle n'a pas trouvé de substitution à une certaine itération)
    #       - On itère sur les mots formés dans notre texte.
    #           - On commencerait par des mots de 2 bytes, puis 3, etc.
    #       - Si le mot n'est pas dans la langue française, on regarde si l'ajout ou la modification d'un byte peut former un mot français.
    #       - Si oui, modifier le byte et vérifier si le texte est plus lisible.
    #       - Voir le google doc, on avait déjà parlé de ça.



    # Écrire decoded_text dans un fichier, en n'oubliant pas de remettre les caractères spéciaux \n, \r, \ufeff.
    decoded_text = ""
    for byte in encoded_bytes:
        if key[byte] is not None:
            decoded_text += key[byte].replace('&', '\n').replace('@', '\r').replace('<', '\ufeff')
        else:
            decoded_text += '~'
    with open("decoded_text.txt", "w", encoding="utf-8") as f:
        f.write(decoded_text)
    return

def verify_substitution_in_text(byte, char, key, text):
    '''
    Cette fonction trouve des mots de 1 à 8 bytes qui contiennent le byte donné et
    vérifie si substituer le byte par le char bénificie au texte.
    '''
    known_chars = set()
    for curr_char in key:
        if curr_char is not None:
            known_chars.add(curr_char)

    unknown_chars = set()
    all_chars = text_to_symbols(text)
    for curr_char in all_chars:
        if not curr_char in known_chars:
            unknown_chars.add(curr_char)

    # On enlève de unknown_chars tous les caractères qui ne sont pas possibles dans les mots français
    global possible_chars_in_word
    unknown_chars = set(curr_char for curr_char in unknown_chars if curr_char in possible_chars_in_word)

    substitution_score = 0
    
    # Trouver tous les groupes de mots de 1 à 8 bytes qui contiennent le byte donné et dont seulement le byte donné est inconnu.
    groups = capture_groups(start_str='[^ ]? [^ ]?', n_bytes_min=1, n_bytes_max=8, end_str='[^ ]? [^ ]?', encoded_bytes=text, key=key, include_start_and_end=True)
    for group in groups:
        # Vérifier si le byte donné est dans le groupe
        if byte in group:
            start = None
            end = None
            if isinstance(group[0], str):
                start = group[0]
            if isinstance(group[-1], str):
                end = group[-1]
            group = [b for b in group if isinstance(b, int)]
            # Tous les autres bytes du groupe devraient avoir une substitution connue dans la clé
            if not all(key[b] is not None for b in group if b != byte):
                continue

            # Si le groupe est un mot français, aucun problème
            if verifier_substitution(group, byte, char, start, end):
                substitution_score += 1
                continue
            # Si le groupe n'est pas un mot français, on doit vérifier s'il y avait un mot français possible avant la substitution.
            # On doit donc vérifier si le groupe est un mot français si on remplace le byte par n'importe quel char de unknown_chars
            char_substitution = verifier_substitutions_possibles(group, byte, possible_chars_in_word, start, end)
            if char_substitution is not None:
                # Il y avait une substitution possible qui formait un mot français, et maintenant il n'y en a plus.
                # La substitution n'est donc pas fiable.
                key[byte] = char
                print("Substitution non fiable de '" + char + "' dans '" + start + ''.join(key[byte] for byte in group) + end + "' car substitution par '" + char_substitution + "' était possible.")
                key[byte] = None
                substitution_score -= 1
            
    # Aucun problème trouvé, la substitution semble fiable.
    return substitution_score >= 0

def verifier_substitution(mot_bytes, byte_a_substituer, caractere_remplacement, start, end):
    """
    Vérifie si la substitution du byte spécifié par un caractère donné
    donne un mot valide selon le dictionnaire.
    """
    # Construire le mot en remplaçant le byte
    mot = "".join(
        key[byte] if byte != byte_a_substituer else caractere_remplacement
        for byte in mot_bytes
    )
    mot = start + mot + end
    mot_espace_1 = mot.index(' ')+1
    mot = mot[mot_espace_1:mot.index(' ', mot_espace_1)]
    # Vérifie si le mot construit est dans le dictionnaire
    if mot in fr_dict:
        print("Mot trouvé: " + mot)
        return True
    return False

def verifier_substitutions_possibles(mot_bytes, byte_a_substituer, caracteres_possibles, start, end):
    """
    Vérifie si la substitution du byte spécifié par un des caractères
    possibles donne un mot du dictionnaire.
    """
    for caractere in caracteres_possibles:
        if verifier_substitution(mot_bytes, byte_a_substituer, caractere, start, end):
            # Retourne le premier caractère valide trouvé
            return caractere
    return None

if __name__ == '__main__':
    for i in range(1):
        print("Test #", i)
        main()
        print()
