import random
from collections import Counter
import requests

# Fonction pour charger le texte

def load_text_from_web(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Une erreur est survenue lors du chargement du texte: {e}")
        return None

# Fonction pour séparer les chaînes de caractères en paires

def cut_string_into_pairs(text):
    pairs = []
    for i in range(0, len(text) - 1, 2):
        pairs.append(text[i:i + 2])
    if len(text) % 2 != 0:
        pairs.append(text[-1] + '_')  # Ajouter un placeholder si le nombre de caractères est impair
    return pairs

# Générer des fréquences à partir du texte

def calculate_frequencies(text):
    # Fréquence des caractères uniques
    char_counter = Counter(text)
    char_probabilities = {char: count / len(text) for char, count in char_counter.items()}
    
    # Fréquence des paires de caractères
    pairs = cut_string_into_pairs(text)
    pair_counter = Counter(pairs)
    pair_probabilities = {pair: count / len(text) for pair, count in pair_counter.items()}
    
    combined_probabilities = char_probabilities.copy()
    combined_probabilities.update(pair_probabilities)
    
    return char_probabilities, pair_probabilities, combined_probabilities

# Génération d'une clé initiale (non optimisée)

def gen_key(symbols):
    if len(symbols) > 256:
        symbols = symbols[:256]  # Limiter à 256 symboles pour éviter les erreurs
    random.seed(1337)
    int_keys = random.sample(list(range(256)), len(symbols))
    key = {s: format(k, '08b') for s, k in zip(symbols, int_keys)}
    return key

# Fonction de score d'évaluation pour une hypothèse de texte décrypté

def eval_score(decrypted_text, french_dict):
    words = decrypted_text.split()
    score = sum(1 for word in words if word in french_dict)
    
    return score

# Fonction pour décrypter un message en utilisant une clé actuelle

def decrypt_message(encoded_message, current_key):
    inverted_key = {v: k for k, v in current_key.items()}  # Inverser la clé pour avoir byte -> symbole
    decoded_chars = []
    for i in range(0, len(encoded_message), 8):
        byte = encoded_message[i:i+8]
        char = inverted_key.get(byte, None)
        if char:
            decoded_chars.append(char)
        else:
            decoded_chars.append('?')  # Ajouter un caractère de remplacement si le byte n'est pas trouvé
    return ''.join(decoded_chars)

# Optimisation de la clé (par permutations)

def optimize_key(encoded_message,key, french_dict, combined_probabilities):
    best_key = key.copy()
    best_score = eval_score(decrypt_message(encoded_message, best_key), french_dict)
    
    improved = True
    iteration = 0
    
    while improved and iteration < 1000:  # Limiter le nombre d'itérations pour éviter les boucles infinies
        improved = False
        # Trier les bytes en fonction des probabilités décroissantes
        sorted_symbols = sorted(combined_probabilities, key=combined_probabilities.get, reverse=True)
        #sorted_symbols2 = sorted(best_key.values() , key=best_key.values() .get, reverse=True)
        
        bytes_char = list(best_key.values())
        
        sorted_bytes = sorted(bytes_char, key=lambda item: combined_probabilities.get(item[0], 0), reverse=True)
        sorted_bytes_values = [item[1] for item in sorted_bytes]
       

        for symbol in bytes_char:
            
          
            if symbol in best_key:
                
                
                continue
            # Trouver tous les caractères/paires de caractères possibles
            
            candidates = combined_probabilities.keys()
            for candidate in candidates:
                if candidate not in best_key:
                    #print(combined_probabilities.keys())

                    temp_key = best_key.copy()
                    temp_key[candidate] = symbol
                    decrypted_text = decrypt_message(encoded_message, temp_key)
                    
                    current_score = eval_score(decrypted_text, french_dict)
                    if current_score > best_score:
                        best_key = temp_key
                        best_score = current_score
                        improved = True
                        break
        iteration += 1
    
    return best_key

# Fonction pour diviser un message en symboles

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

# Fonction pour chiffrer un message

def chiffrer(M, K):
    l = M_vers_symboles(M, K)
    l = [K[x] for x in l]
    return ''.join(l)

# Fonction pour calculer les probabilités pour une clé donnée

def calculate_key_probabilities(key, corpus):
    # Inverser la clé pour obtenir byte -> symbole
    inverted_key = {v: k for k, v in key.items()}
    symbol_counts = Counter(inverted_key.values())
    total_count = sum(symbol_counts.values())
    key_probabilities = {symbol: count / total_count for symbol, count in symbol_counts.items()}
    return key_probabilities

# Fonction pour écrire des probabilités dans un fichier

def write_probabilities_to_file(filename, probabilities):
    with open(filename, 'w', encoding='utf-8') as f:
        for symbol, probability in probabilities.items():
            f.write(f"{symbol}: {probability}\n")

# Exécution du décryptage
if __name__ == "__main__":
    # Charger le corpus de texte
    url_1 = "https://www.gutenberg.org/ebooks/13846.txt.utf-8"
    url_2 = "https://www.gutenberg.org/ebooks/4650.txt.utf-8"
    corpus = load_text_from_web(url_1) + load_text_from_web(url_2)

    if corpus is None:
        print("Impossible de charger le corpus de texte. Veuillez vérifier les URLs.")
    else:
        # Calculer les fréquences des caractères et des paires
        char_probabilities, pair_probabilities, combined_probabilities = calculate_frequencies(corpus)

        # Créer une clé initiale pour chiffrer le message
        all_symbols = list(set(corpus)) + cut_string_into_pairs(corpus)
 # Limiter à 256 symboles pour éviter les erreurs de clé
        initial_key = gen_key(all_symbols)

        # Charger le dictionnaire français depuis le repo 
        url_dict = "https://raw.githubusercontent.com/Taknok/French-Wordlist/master/francais.txt"
        french_dict_text = load_text_from_web(url_dict)
        if french_dict_text is None:
            print("Impossible de charger le dictionnaire français. Veuillez vérifier l'URL du dictionnaire.")
        else:
            french_dict = set(word.strip().lower() for word in french_dict_text.splitlines())

            # Exemple de message à chiffrer
            M = corpus[10000:10100]
            C = chiffrer(M, initial_key)

            print(f"M = \"{M}\"")
            print(f"\nLongueur du message M = {len(M)}")
            print(f"\nDivision en symboles = {M_vers_symboles(M, initial_key)}")
            print(f"\nNombre de symboles = {len(M_vers_symboles(M, initial_key))}")
            print(f"\nC = \"{C}\"")
            print(f"\nLongueur du cryptogramme C en bits = {len(C)}")
            print(f"\nLongueur du cryptogramme C en octets = {len(C) // 8}")

            # Générer une clé différente pour casser le chiffrement (random_key)
            random_text = corpus[20000:30000]  # Utiliser une partie différente du texte pour générer la clé
            caracteres = list(set(list(random_text)))
            nb_caracteres = len(caracteres)
            nb_bicaracteres = 256-nb_caracteres
            all_symbols_random = caracteres + [item for item, _ in Counter(cut_string_into_pairs(random_text)).most_common(nb_bicaracteres)]# Sélectionner les symboles
            random_key = gen_key(all_symbols_random) # gen la key
             


            optimized_key = optimize_key(C, random_key, french_dict, combined_probabilities)

            # Calculer les probabilités des clés (initiale et optimisée)
            initial_key_probabilities = calculate_key_probabilities(initial_key, corpus)
            optimized_key_probabilities = calculate_key_probabilities(optimized_key, corpus)

            # Écrire les probabilités dans des fichiers
            write_probabilities_to_file("initial_key_probabilities.txt", initial_key_probabilities)
            write_probabilities_to_file("optimized_key_probabilities.txt", optimized_key_probabilities)

            # Déchiffrer le message avec la clé optimisée
            decrypted_message = decrypt_message(C, optimized_key)
            print(f"\nMessage déchiffré : {decrypted_message}")
